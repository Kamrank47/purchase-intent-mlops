# Security policy

## Reporting a vulnerability

Email **kamranali.k47@gmail.com** with a description, the affected file or endpoint,
and reproduction steps. Please do not open a public issue for a suspected
vulnerability. Expect an acknowledgement within 72 hours.

## Scope

- The inference API in `src/intent_mlops/serve.py`
- The container image and Kubernetes manifests
- CI and release workflows

## Posture

- No secrets in the repository; CI uses `GITHUB_TOKEN` and the optional MLflow
  server runs locally.
- The container runs as a non-root user with a read-only root filesystem and all
  Linux capabilities dropped in Kubernetes.
- Dependencies are pinned by the lockless `pyproject.toml` constraints, updated
  weekly by Dependabot, and the image is scanned with Trivy in CI.
- Input validation is enforced by Pydantic, including ranges and allowed categories.
- The model card documents intended use and known limitations.
