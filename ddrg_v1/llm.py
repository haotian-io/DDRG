"""LLM adapter layer for DDRG v1.

Only this file knows how to call a model provider. The pipeline calls
`LLMClient.generate(...)`, so changing platforms or adding local deployment
support should not require editing the DDRG method code.
"""

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Protocol

from openai import AzureOpenAI, OpenAI


class LLMClient(Protocol):
    def generate(
        self,
        content: str,
        model: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> str:
        """Generate text for a single user message."""


def _iter_env_files() -> Iterable[Path]:
    here = Path(__file__).resolve()
    seen: set[Path] = set()
    for parent in here.parents:
        env_path = parent / ".env"
        if env_path in seen:
            continue
        seen.add(env_path)
        yield env_path


def _load_dotenv_if_present() -> None:
    for env_path in _iter_env_files():
        if not env_path.exists():
            continue
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not key or key in os.environ:
                continue
            value = value.strip().strip('"').strip("'")
            os.environ[key] = value
        return


def _pick_env(*names: str) -> Optional[str]:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


@dataclass
class OpenAICompatibleClient:
    base_url: str
    api_key: str
    referer: str = "https://localhost"
    title: str = "ddrg-v1"
    retries: int = 3
    retry_sleep: float = 2.0

    def __post_init__(self) -> None:
        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def generate(
        self,
        content: str,
        model: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> str:
        for attempt in range(self.retries + 1):
            try:
                completion = self._client.chat.completions.create(
                    extra_headers={
                        "HTTP-Referer": self.referer,
                        "X-Title": self.title,
                    },
                    model=model,
                    messages=[{"role": "user", "content": content}],
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                )
                return completion.choices[0].message.content or ""
            except Exception as exc:
                if attempt >= self.retries:
                    raise RuntimeError(f"API call failed: {exc}") from exc
                time.sleep(self.retry_sleep * (2**attempt))
        return ""


@dataclass
class AzureOpenAIClient:
    azure_endpoint: str
    api_key: str
    api_version: str
    azure_deployment: Optional[str] = None
    retries: int = 3
    retry_sleep: float = 2.0

    def __post_init__(self) -> None:
        self._client = AzureOpenAI(
            azure_endpoint=self.azure_endpoint,
            api_key=self.api_key,
            api_version=self.api_version,
        )

    def generate(
        self,
        content: str,
        model: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> str:
        deployment = self.azure_deployment or model
        if not deployment:
            raise RuntimeError("Azure OpenAI requires a deployment/model name.")
        for attempt in range(self.retries + 1):
            try:
                completion = self._client.chat.completions.create(
                    model=deployment,
                    messages=[{"role": "user", "content": content}],
                    temperature=temperature,
                    top_p=top_p,
                    max_completion_tokens=max_tokens,
                )
                return completion.choices[0].message.content or ""
            except Exception as exc:
                if attempt >= self.retries:
                    raise RuntimeError(f"Azure OpenAI API call failed: {exc}") from exc
                time.sleep(self.retry_sleep * (2**attempt))
        return ""


def make_llm_client(
    provider: str = "openai-compatible",
    base_url: str = "https://openrouter.ai/api/v1",
    api_key: Optional[str] = None,
    azure_endpoint: Optional[str] = None,
    azure_api_version: Optional[str] = None,
    azure_deployment: Optional[str] = None,
    referer: str = "https://localhost",
    title: str = "ddrg-v1",
    retries: int = 3,
    retry_sleep: float = 2.0,
) -> LLMClient:
    _load_dotenv_if_present()
    provider = provider.strip().lower()
    if provider in {"openai-compatible", "openai", "openrouter"}:
        key = api_key or _pick_env("OPENROUTER_API_KEY", "OPENAI_API_KEY")
        if not key:
            raise RuntimeError("Set OPENROUTER_API_KEY/OPENAI_API_KEY or pass --api-key.")
        return OpenAICompatibleClient(
            base_url=base_url,
            api_key=key,
            referer=referer,
            title=title,
            retries=retries,
            retry_sleep=retry_sleep,
        )
    if provider in {"azure", "azure-openai", "azure_openai"}:
        key = api_key or _pick_env("AZURE_OPENAI_API_KEY")
        endpoint = azure_endpoint or _pick_env("AZURE_OPENAI_ENDPOINT")
        api_version = azure_api_version or _pick_env("AZURE_OPENAI_API_VERSION")
        deployment = azure_deployment or _pick_env("AZURE_OPENAI_DEPLOYMENT")
        missing: Dict[str, Optional[str]] = {
            "AZURE_OPENAI_API_KEY": key,
            "AZURE_OPENAI_ENDPOINT": endpoint,
            "AZURE_OPENAI_API_VERSION": api_version,
        }
        missing_names = [name for name, value in missing.items() if not value]
        if missing_names:
            missing_str = ", ".join(missing_names)
            raise RuntimeError(
                f"Set {missing_str} or pass --api-key/--azure-endpoint/--azure-api-version."
            )
        return AzureOpenAIClient(
            azure_endpoint=endpoint or "",
            api_key=key or "",
            api_version=api_version or "",
            azure_deployment=deployment,
            retries=retries,
            retry_sleep=retry_sleep,
        )
    raise ValueError(f"Unsupported LLM provider: {provider}")


def prompt_with_problem(prompt: str, question: str) -> str:
    return f"{prompt}\n\nProblem:\n{question}"
