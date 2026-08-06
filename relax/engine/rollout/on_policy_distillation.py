# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
import json
from collections.abc import Awaitable, Callable, Sequence
from time import monotonic

import aiohttp
import numpy as np

from relax.utils.logging_utils import get_logger
from relax.utils.opd import opd_main_worker, opd_opsd_worker
from relax.utils.opd.opd_utils import is_sdpo_prompt_routing_enabled
from relax.utils.opd.sdpo import SDPO_TOKEN_SELECTION, validate_sdpo_text_only
from relax.utils.types import Sample


try:
    import orjson

    def _dumps_to_bytes(payload: dict) -> bytes:
        return orjson.dumps(payload, option=orjson.OPT_SERIALIZE_NUMPY)
except ImportError:  # pragma: no cover - orjson is normally available via sglang

    def _dumps_to_bytes(payload: dict) -> bytes:
        return json.dumps(payload).encode("utf-8")


logger = get_logger(__name__)

EncodeMultimodalInputs = Callable[[dict], Awaitable[tuple[dict, float]]]


def _has_sdpo_teacher_prompt(prompt: object) -> bool:
    if isinstance(prompt, str):
        return bool(prompt.strip())
    if isinstance(prompt, list):
        return bool(prompt)
    return False


def _aiohttp_json_post_kwargs(payload: dict) -> dict:
    """Bypass aiohttp's ``json=`` kwarg: serialize via orjson and ship as raw
    ``data=``."""
    return {
        "data": _dumps_to_bytes(payload),
        "headers": {"Content-Type": "application/json"},
    }


def is_opd_enabled(args, evaluation: bool = False) -> bool:
    return not evaluation and getattr(args, "use_opd", False) and getattr(args, "opd_type", None) == "sglang"


def _create_teacher_client_session(args) -> aiohttp.ClientSession:
    connector_limit = int(getattr(args, "opd_teacher_connector_limit", 256))
    connector = aiohttp.TCPConnector(limit=connector_limit)
    timeout = aiohttp.ClientTimeout(total=float(args.opd_teacher_timeout_s))
    return aiohttp.ClientSession(connector=connector, timeout=timeout)


# --- Teacher URL selection: MOPD routing (by data_source) x replica round-robin ---
# Two layers:
#   1. Routing (MOPD): when ``args.opd_teacher_routes_map`` is set, pick the
#      teacher for this sample by ``sample.metadata[args.opd_teacher_key]``.
#      Each route value is a LIST of that teacher's replica URLs.
#   2. Replica round-robin: spread requests across a teacher's replicas with a
#      per-teacher in-process counter (single-threaded asyncio makes ``+= 1`` safe).
# Single-teacher path falls back to ``args.opd_teacher_urls`` / ``opd_teacher_url``.
_TEACHER_URL_RR: dict[str, int] = {}


def _round_robin(urls: list[str], key: str) -> str:
    i = _TEACHER_URL_RR.get(key, 0)
    _TEACHER_URL_RR[key] = i + 1
    return urls[i % len(urls)]


def _pick_teacher_url(args, sample=None) -> str:
    routes_map = getattr(args, "opd_teacher_routes_map", None)
    if routes_map and sample is not None:
        key_field = getattr(args, "opd_teacher_key", None) or "data_source"
        metadata = getattr(sample, "metadata", None) or {}
        routing_value = metadata.get(key_field)
        if routing_value is None:
            raise ValueError(
                f"MOPD routing: sample missing key '{key_field}' in metadata. "
                f"Available metadata keys: {list(metadata.keys())}. "
                f"Ensure the dataset has a '{key_field}' column and it is surfaced "
                "via --metadata-key or the data pipeline."
            )
        replicas = routes_map.get(routing_value)
        if not replicas:
            raise KeyError(
                f"MOPD routing: no teacher route for '{key_field}={routing_value}'. "
                f"Available routes: {list(routes_map.keys())}."
            )
        return _round_robin(replicas, routing_value)
    # Single-teacher path: round-robin over replicas if configured.
    urls = getattr(args, "opd_teacher_urls", None)
    if urls and len(urls) > 1:
        return _round_robin(urls, "__single__")
    return args.opd_teacher_url


class OpdManager:
    def __init__(self, args):
        self.args = args
        self.is_sdpo = is_sdpo_prompt_routing_enabled(args)
        self.topk_worker: opd_main_worker.TopkWorker | None = None
        self.sampled_worker: opd_main_worker.SampledTokenWorker | None = None  # 仅 student_sampled
        opsd_worker = opd_opsd_worker.OpsdWorker.from_args(args)
        self.opsd_worker = opsd_worker if opsd_worker.is_opsd else None
        if self.is_sdpo and self.opsd_worker is None:
            self.opsd_worker = opd_opsd_worker.OpsdWorker(is_opsd=True)

        token_selection = args.opd_token_selection
        if token_selection != "student_sampled":
            self.topk_worker = opd_main_worker.TopkWorker.from_args(args)
        else:
            self.sampled_worker = opd_main_worker.SampledTokenWorker.from_args(args)

    @property
    def is_topk(self) -> bool:
        return self.topk_worker is not None

    @property
    def is_opsd(self) -> bool:
        return self.opsd_worker is not None

    def _validate_sdpo_configuration(self) -> None:
        if not self.is_sdpo:
            return
        if self.topk_worker is None or self.topk_worker.spec.name != SDPO_TOKEN_SELECTION:
            raise ValueError(
                "SDPO only supports student_topk token selection; "
                "set --opd-token-selection student_topk and --opd-log-prob-top-k > 0."
            )

    def schema_opd_transfer_data(self) -> list[str]:
        fields: list[str] = []
        if self.topk_worker is not None:
            fields.extend(self.topk_worker.topk_transfer_fields())
        if self.sampled_worker is not None:
            fields.extend(self.sampled_worker.sampled_transfer_fields())
        return fields

    @staticmethod
    def _clear_teacher_payload(sample: Sample) -> None:
        """Drop outputs from an earlier teacher request without touching
        rollout Top-K."""

        for field_name in (
            "teacher_log_probs",
            "teacher_topk_token_ids",
            "teacher_topk_log_probs",
            "teacher_at_student_topk_log_probs",
            "student_at_teacher_topk_log_probs",
            "opd_topk_token_ids",
            "opd_topk_student_log_probs",
            "opd_topk_teacher_log_probs",
            "opd_topk_ksz",
            "teacher_tokens",
            "teacher_prompt_length",
        ):
            setattr(sample, field_name, None)

    def produce_opd_transfer_data(self, samples: list[Sample], train_data: dict) -> None:
        if self.topk_worker is not None:
            transfer_fields = (
                self.topk_worker.topk_transfer_fields() if self.is_sdpo else opd_main_worker.TopkWorker.TRANSFER_FIELDS
            )
            for field_name in transfer_fields:
                if not any(getattr(s, field_name, None) is not None for s in samples):
                    continue
                flat: list = []
                for s in samples:
                    v = getattr(s, field_name, None)
                    if v is None:
                        flat.append([])
                    else:
                        flat.append(v.reshape(-1).tolist())
                train_data[field_name] = flat
            kl_field = opd_main_worker.TopkWorker.TRANSFER_K_LENGTHS
            if self.topk_worker.spec.name == "union" and any(getattr(s, kl_field, None) is not None for s in samples):
                train_data[kl_field] = [
                    getattr(s, kl_field).tolist() if getattr(s, kl_field, None) is not None else [] for s in samples
                ]
        elif self.sampled_worker is not None:
            train_data[opd_main_worker.SampledTokenWorker.TRANSFER_TEACHER_LOG_PROBS] = [
                s.teacher_log_probs if s.teacher_log_probs is not None else [] for s in samples
            ]

    def before_rollout(self, payload: dict) -> None:
        if self.topk_worker is None:
            return
        fields = self.topk_worker.student_rollout_payload()
        if fields:
            payload.update(fields)

    def parse_rollout_logprobs(self, meta_info: dict, tokens: list, log_probs: list) -> tuple[list, list]:
        if self.topk_worker is None:
            return tokens, log_probs
        val_b64 = meta_info.get("output_token_logprobs_val_b64")
        if val_b64 is None:
            return tokens, log_probs
        import numpy as np

        try:
            import pybase64
        except ImportError:  # pragma: no cover - pybase64 is used in the runtime image
            import base64 as pybase64

        val = np.frombuffer(pybase64.b64decode(val_b64), dtype=np.float32)
        idx_b64 = meta_info.get("output_token_logprobs_idx_b64")
        idx = np.frombuffer(pybase64.b64decode(idx_b64), dtype=np.int32) if idx_b64 else np.array([], dtype=np.int32)
        n_expected = len(tokens)
        if idx.size == n_expected and val.size == n_expected:
            out_tokens = idx.tolist()
            out_log_probs = val.tolist()
        else:
            out_tokens = tokens
            out_log_probs = log_probs
        return out_tokens, out_log_probs

    def after_rollout(self, sample: Sample, output: dict) -> None:
        if self.topk_worker is None:
            return
        pair = opd_main_worker.LogprobResponse(output).self_topk("rollout", self.topk_worker.top_k)
        if pair is None:
            return
        token_ids, log_probs = pair
        if sample.student_topk_token_ids is None:
            sample.student_topk_token_ids = token_ids
            sample.student_topk_log_probs = log_probs
        else:  # multi-turn
            sample.student_topk_token_ids = np.vstack([sample.student_topk_token_ids, token_ids])
            sample.student_topk_log_probs = np.vstack([sample.student_topk_log_probs, log_probs])

    async def prefill(
        self,
        samples: Sample | Sequence[Sample],
        encode_multimodal_inputs: EncodeMultimodalInputs | None = None,
    ) -> None:
        sample_list = list(samples) if isinstance(samples, Sequence) else [samples]
        self._validate_sdpo_configuration()
        if self.is_sdpo:
            for sample in sample_list:
                validate_sdpo_text_only(sample)
            for sample in sample_list:
                self._clear_teacher_payload(sample)

        opsd_worker = getattr(self, "opsd_worker", None)
        if opsd_worker is not None:
            await asyncio.gather(*[opsd_worker.build_teacher_inputs(self.args, sample) for sample in sample_list])

        prefill_start = monotonic()
        if self.is_sdpo:
            request_indices = [i for i, sample in enumerate(sample_list) if self._needs_teacher_request(sample)]
        else:
            request_indices = list(range(len(sample_list)))
        fetch_results: list[bool | None] = [None] * len(sample_list)
        if request_indices:
            async with _create_teacher_client_session(self.args) as session:
                requested_results = await asyncio.gather(
                    *[self._teacher_prefill(sample_list[i], session) for i in request_indices]
                )
                for index, result in zip(request_indices, requested_results, strict=True):
                    fetch_results[index] = result
                self._raise_if_all_failed(
                    [sample_list[i] for i in request_indices],
                    requested_results,
                )

                if self.topk_worker is not None and self.topk_worker.spec.student_at_teacher:
                    await asyncio.gather(
                        *[
                            self._student_prefill(sample_list[i], session, encode_multimodal_inputs)
                            for i in request_indices
                        ]
                    )

        self._assemble_transfer(sample_list)
        if self.is_sdpo:
            logger.info(
                "SDPO teacher prefill: requests=%d valid=%d elapsed=%.3fs",
                len(request_indices),
                sum(int(result is True) for result in fetch_results),
                monotonic() - prefill_start,
            )

    def _needs_teacher_request(self, sample: Sample) -> bool:
        response_length = int(sample.response_length or 0)
        if response_length <= 0:
            return False
        if self.is_sdpo and not _has_sdpo_teacher_prompt(sample.teacher_prompt):
            raise ValueError(
                f"SDPO requires a teacher prompt for every non-empty response; sample_index={sample.index}"
            )
        return True

    async def _post_logprob(
        self,
        session: aiohttp.ClientSession,
        url: str,
        payload: dict,
        sample: Sample,
        err_tag: str,
    ) -> opd_main_worker.LogprobResponse | None:
        try:
            async with session.post(url, **_aiohttp_json_post_kwargs(payload)) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    logger.error("OPD %s failed: status=%s, url=%s, body=%s", err_tag, resp.status, url, body[:2048])
                    resp.raise_for_status()
                data = await resp.json()
        except Exception as exc:
            logger.error(
                "OPD %s fetch failed for sample_index=%s, error=%s",
                err_tag,
                getattr(sample, "index", None),
                f"{type(exc).__name__}: {str(exc)[:256]}",
            )
            if self.is_sdpo:
                raise RuntimeError(f"SDPO {err_tag} failed for sample_index={getattr(sample, 'index', None)}") from exc
            return None

        return opd_main_worker.LogprobResponse(data)

    async def _teacher_prefill(self, sample: Sample, session: aiohttp.ClientSession) -> bool:
        response_length = int(sample.response_length or 0)
        if response_length <= 0:
            return True
        if self.is_sdpo and not _has_sdpo_teacher_prompt(sample.teacher_prompt):
            raise ValueError(
                f"SDPO requires a teacher prompt for every non-empty response; sample_index={sample.index}"
            )

        opsd_worker = getattr(self, "opsd_worker", None)
        if opsd_worker is not None:
            image_data = opsd_worker.build_preexpanded_image_data(sample)
            teacher_input_ids = opsd_worker.teacher_input_ids(sample, response_length)
            prompt_length = opsd_worker.teacher_prompt_len(sample, response_length)
            logprob_start_len = max(prompt_length - 1, 0)
        else:
            image_data = None
            teacher_input_ids = sample.rollout_tokens or sample.tokens
            logprob_start_len = max(len(sample.tokens) - response_length - 1, 0)

        mm_fields = {"image_data": image_data} if image_data is not None else None
        if self.topk_worker is not None:
            if self.is_sdpo:
                from relax.utils.opd.sdpo import validate_sdpo_student_topk_ids

                validate_sdpo_student_topk_ids(
                    token_ids=sample.student_topk_token_ids,
                    response_rows=response_length,
                    top_k=self.topk_worker.top_k,
                    sample_index=int(sample.index) if sample.index is not None else -1,
                )
            payload = self.topk_worker.build_teacher_payload(
                input_ids=teacher_input_ids,
                logprob_start_len=logprob_start_len,
                student_topk_ids=sample.student_topk_token_ids,
                response_length=response_length,
                mm_fields=mm_fields,
            )
        else:
            payload = opd_main_worker.build_prefill_payload_base(teacher_input_ids, logprob_start_len)
            if mm_fields:
                payload.update(mm_fields)

        teacher_url = _pick_teacher_url(self.args, sample)
        resp_obj = await self._post_logprob(session, teacher_url, payload, sample, "teacher prefill")
        if resp_obj is None:
            if self.is_sdpo:
                raise RuntimeError(f"SDPO teacher response is empty for sample_index={getattr(sample, 'index', None)}")
            return False

        token_logprobs = resp_obj.base_logprobs_1d()
        if token_logprobs is None or len(token_logprobs) == 0:
            logger.error(
                "Invalid OPD teacher response for sample_index=%s: missing input_token_logprobs.",
                getattr(sample, "index", None),
            )
            if self.is_sdpo:
                raise ValueError(
                    "SDPO teacher response is missing input_token_logprobs for "
                    f"sample_index={getattr(sample, 'index', None)}"
                )
            return False

        if len(token_logprobs) != response_length + 1:
            logger.error(
                "Teacher log-prob length mismatch for sample_index=%s: got=%s expected=%s",
                getattr(sample, "index", None),
                len(token_logprobs) - 1,
                response_length,
            )
            if self.is_sdpo:
                raise ValueError(
                    "SDPO teacher log-prob length mismatch for "
                    f"sample_index={getattr(sample, 'index', None)}: "
                    f"got={len(token_logprobs) - 1}, expected={response_length}"
                )
            return False

        if self.sampled_worker is not None:
            sample.teacher_log_probs = [float(v) for v in token_logprobs[1 : 1 + response_length]]

        if self.topk_worker is not None:
            if self.topk_worker.spec.teacher_self_topk:
                pair = self.topk_worker.parse_prefill_self_topk(resp_obj, response_length)
                sample.teacher_topk_token_ids = pair[0] if pair else None
                sample.teacher_topk_log_probs = pair[1] if pair else None
            if self.topk_worker.spec.teacher_at_student:
                sample.teacher_at_student_topk_log_probs = self.topk_worker.parse_prefill_other_topk(
                    resp_obj, response_length
                )
                if self.is_sdpo and sample.teacher_at_student_topk_log_probs is None:
                    raise ValueError(
                        "SDPO teacher-on-student Top-K payload is missing for "
                        f"sample_index={getattr(sample, 'index', None)}"
                    )

        return True

    async def _student_prefill(
        self, sample: Sample, session: aiohttp.ClientSession, encode_mm_fn: EncodeMultimodalInputs | None
    ) -> None:

        from relax.utils.opd.opd_utils import build_student_preexpanded_image_data

        response_length = int(sample.response_length or 0)
        teacher_topk_ids = sample.teacher_topk_token_ids

        prompt_length = len(sample.tokens) - response_length
        logprob_start_len = max(prompt_length - 1, 0)

        preexpanded_image_data = await build_student_preexpanded_image_data(sample)
        if preexpanded_image_data is not None:
            student_input_ids = sample.tokens
            mm_fields = {"image_data": preexpanded_image_data}
        else:
            student_input_ids = sample.rollout_tokens or sample.tokens
            mm_fields = None
            if encode_mm_fn is not None and sample.multimodal_inputs and sample.multimodal_inputs.get("images"):
                mm_fields, _ = await encode_mm_fn(sample.multimodal_inputs)

        payload = self.topk_worker.build_student_payload(
            input_ids=student_input_ids,
            logprob_start_len=logprob_start_len,
            teacher_topk_ids=teacher_topk_ids,
            response_length=response_length,
            mm_fields=mm_fields,
        )

        student_url = f"http://{self.args.sglang_router_ip}:{self.args.sglang_router_port}/generate"
        resp_obj = await self._post_logprob(session, student_url, payload, sample, "student-at-teacher-topk")
        if resp_obj is None:
            if self.is_sdpo:
                raise RuntimeError(
                    f"SDPO student-at-teacher-topk response is empty for sample_index={getattr(sample, 'index', None)}"
                )
            return
        sample.student_at_teacher_topk_log_probs = self.topk_worker.parse_prefill_other_topk(resp_obj, response_length)
        if self.is_sdpo and sample.student_at_teacher_topk_log_probs is None:
            raise ValueError(
                f"SDPO student-at-teacher-topk payload is missing for sample_index={getattr(sample, 'index', None)}"
            )

    def _assemble_transfer(self, samples: list[Sample]) -> None:
        if self.topk_worker is None:
            return
        for sample in samples:
            student_self = (
                (sample.student_topk_token_ids, sample.student_topk_log_probs)
                if sample.student_topk_token_ids is not None
                else None
            )
            teacher_self = (
                (sample.teacher_topk_token_ids, sample.teacher_topk_log_probs)
                if sample.teacher_topk_token_ids is not None
                else None
            )
            channels = self.topk_worker.build_transfer_channels(
                student_self_topk=student_self,
                teacher_self_topk=teacher_self,
                teacher_at_student_lp=sample.teacher_at_student_topk_log_probs,
                student_at_teacher_lp=sample.student_at_teacher_topk_log_probs,
            )
            if self.is_sdpo and int(sample.response_length or 0) > 0:
                from relax.utils.opd.sdpo import validate_sdpo_topk_payload

                validate_sdpo_topk_payload(
                    token_ids=channels.get(opd_main_worker.TopkWorker.TRANSFER_TOKEN_IDS),
                    teacher_log_probs=channels.get(opd_main_worker.TopkWorker.TRANSFER_TEACHER_LOG_PROBS),
                    response_rows=int(sample.response_length),
                    top_k=self.topk_worker.top_k,
                    sample_index=int(sample.index) if sample.index is not None else -1,
                )
            sample.opd_topk_token_ids = channels.get(opd_main_worker.TopkWorker.TRANSFER_TOKEN_IDS)
            sample.opd_topk_student_log_probs = channels.get(opd_main_worker.TopkWorker.TRANSFER_STUDENT_LOG_PROBS)
            sample.opd_topk_teacher_log_probs = channels.get(opd_main_worker.TopkWorker.TRANSFER_TEACHER_LOG_PROBS)
            sample.opd_topk_ksz = channels.get(opd_main_worker.TopkWorker.TRANSFER_K_LENGTHS)

    @staticmethod
    def _raise_if_all_failed(samples: list[Sample], fetch_results: list[bool]) -> None:
        eligible = [(s, ok) for s, ok in zip(samples, fetch_results) if int(getattr(s, "response_length", 0) or 0) > 0]
        if not eligible or any(ok for _, ok in eligible):
            return
        raise RuntimeError(
            f"All OPD teacher fetches failed for {len(eligible)} non-empty samples "
            f"(total={len(samples)}); url={getattr(samples[0], 'opd_teacher_url', None) if samples else None}"
        )


def produce_opd_transfer_data(args, samples: list[Sample], train_data: dict) -> None:
    if not is_opd_enabled(args):
        return
    OpdManager(args).produce_opd_transfer_data(samples, train_data)
