"""Offline request-boundary tests using the repository-pinned SDK."""
import json
import os
import unittest
from unittest.mock import patch

import httpx
from openai import OpenAI

import run


class ApiRequestTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.requests = []

    def respond(self, request):
        self.requests.append(request)
        return httpx.Response(200, json={
            "id": "offline-response", "object": "chat.completion", "created": 0,
            "model": json.loads(request.content)["model"],
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": "Thought: Ready. Action: wait",
                "reasoning_content": "Preserve this reasoning on the next turn.",
            }}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        })

    def request(self, argv, messages=None):
        args = run.parse_args(["--api_key", "offline-fixture", *argv])
        if messages is None:
            if args.text_only:
                observation = run.format_msg_text_only(1, "Task", "", "", "Tree")
            else:
                observation = run.format_msg(1, "Task", "", "", "cG5n", "Page")
            messages = [{"role": "system", "content": "Follow instructions."}, observation]
        with httpx.Client(transport=httpx.MockTransport(self.respond)) as transport:
            with patch.object(run, "OpenAI", side_effect=lambda **kwargs: OpenAI(
                    **kwargs, http_client=transport, max_retries=0)):
                client = run.build_openai_client(args)
            result = run.call_gpt4v_api(args, client, messages)
        self.assertFalse(result[2])
        self.assertEqual(result[:2], (10, 5))
        payload = json.loads(self.requests[-1].content)
        self.assertEqual(payload["messages"], messages)
        self.assertEqual(self.requests[-1].method, "POST")
        for unsupported in ("thinking", "reasoning_effort", "reasoning_split", "service_tier"):
            self.assertNotIn(unsupported, payload)
        return args, payload, result[3]

    def test_region_defaults_and_both_observation_modes(self):
        for provider, host in (("minimax", "api.minimax.io"), ("minimax_cn", "api.minimaxi.com")):
            for mode in ([], ["--text_only"]):
                with self.subTest(provider=provider, mode=mode):
                    _, body, _ = self.request(["--api_provider", provider, *mode])
                    self.assertEqual(str(self.requests[-1].url), f"https://{host}/v1/chat/completions")
                    self.assertEqual(body["model"], "MiniMax-M3.1-Flash")
                    self.assertEqual(body["max_tokens"], 32768)
                    content = body["messages"][-1]["content"]
                    if mode:
                        self.assertIsInstance(content, str)
                        self.assertEqual(self.requests[-1].extensions["timeout"]["read"], 30)
                    else:
                        self.assertEqual(content[1]["image_url"]["url"], "data:image/png;base64,cG5n")

    def test_explicit_current_legacy_and_custom_models(self):
        for model in ("MiniMax-M3", "MiniMax-M2.7", "custom-saved-model", "gpt-4-vision-preview"):
            for provider in ("minimax", "minimax_cn"):
                with self.subTest(model=model, provider=provider):
                    _, body, _ = self.request(["--api_provider", provider, "--api_model", model,
                                               "--max_tokens", "2048", "--seed", "42", "--text_only"])
                    self.assertEqual(body["model"], model)
                    self.assertEqual(body["max_tokens"], 2048)
                    self.assertEqual(body["seed"], 42)

    def test_base_url_and_environment_defaults(self):
        for base_url in ("https://api.minimax.io/v1/", "https://api.minimaxi.com/v1"):
            for environment in (False, True):
                with self.subTest(base_url=base_url, environment=environment):
                    with patch.dict(os.environ, {"OPENAI_BASE_URL": base_url} if environment else {}):
                        argv = [] if environment else ["--api_base_url", base_url]
                        _, body, _ = self.request(argv)
                    self.assertEqual(body["model"], "MiniMax-M3.1-Flash")
                    self.assertEqual(str(self.requests[-1].url), base_url.rstrip("/") + "/chat/completions")

    def test_custom_endpoint_precedence_and_original_defaults(self):
        _, body, _ = self.request([])
        self.assertEqual(body["model"], "gpt-4-vision-preview")
        self.assertEqual(body["max_tokens"], 1000)
        self.assertEqual(str(self.requests[-1].url), "https://api.openai.com/v1/chat/completions")
        for model in (None, "saved-model"):
            with self.subTest(model=model), patch.dict(os.environ, {"OPENAI_BASE_URL": "https://api.minimax.io/v1"}):
                argv = ["--api_provider", "minimax_cn", "--api_base_url", "https://gateway.invalid/v1"]
                if model:
                    argv += ["--api_model", model]
                _, body, _ = self.request(argv)
                self.assertEqual(body["model"], model or "gpt-4-vision-preview")
                self.assertEqual(body["max_tokens"], 1000)
                self.assertEqual(str(self.requests[-1].url), "https://gateway.invalid/v1/chat/completions")

    def test_reasoning_and_tool_history_round_trip(self):
        argv = ["--api_provider", "minimax", "--text_only"]
        _, _, response = self.request(argv)
        messages = [{"role": "user", "content": "First turn"}]
        self.assertEqual(run.append_assistant_response(messages, response), "Thought: Ready. Action: wait")
        self.assertEqual(messages[-1]["reasoning_content"], "Preserve this reasoning on the next turn.")
        messages.extend([
            {"role": "assistant", "content": "<think>Keep this too.</think>", "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "inspect", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "Page contents"},
            {"role": "user", "content": "Continue"},
        ])
        for clip in (run.clip_message_and_obs, run.clip_message_and_obs_text_only):
            with self.subTest(clip=clip.__name__):
                clipped = clip(messages, 1)
                self.assertEqual(clipped[1:4], messages[1:4])
                self.request(argv, clipped)

    def test_main_uses_resolved_arguments(self):
        class StopBeforeBrowser(Exception):
            pass
        with patch("sys.argv", ["run.py", "--api_provider", "minimax", "--text_only"]), \
                patch.object(run, "build_openai_client") as build, \
                patch.object(run, "driver_config", side_effect=StopBeforeBrowser):
            with self.assertRaises(StopBeforeBrowser):
                run.main()
        args = build.call_args.args[0]
        self.assertEqual(args.api_model, "MiniMax-M3.1-Flash")
        self.assertEqual(args.max_tokens, 32768)


if __name__ == "__main__":
    unittest.main()
