"""Helpers for OpenAI-compatible API endpoint URLs."""


KNOWN_ENDPOINTS = ("chat/completions", "embeddings", "responses")


def endpoint_url(base_or_endpoint_url: str, endpoint: str) -> str:
    """Return a normalized URL for an OpenAI-compatible endpoint."""

    clean_url = base_or_endpoint_url.strip().rstrip("/")
    clean_endpoint = endpoint.strip("/")

    if not clean_url:
        raise ValueError("API URL cannot be empty")

    if clean_url.endswith(f"/{clean_endpoint}"):
        return clean_url

    for known_endpoint in KNOWN_ENDPOINTS:
        if clean_url.endswith(f"/{known_endpoint}"):
            base_url = clean_url[: -len(known_endpoint)].rstrip("/")
            return f"{base_url}/{clean_endpoint}"

    if clean_url.endswith("/v1"):
        return f"{clean_url}/{clean_endpoint}"

    return f"{clean_url}/v1/{clean_endpoint}"
