"""Build the dependency-free AWS Lambda artifact; does not create paid resources."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


def build(out):
    root = Path(__file__).resolve().parents[1]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    sources = {"handler.py": root / "deploy/aws/handler.py", "connector_protocol.py": root / "src/gaptrace/connector_protocol.py",
               "literature.py": root / "src/gaptrace/literature.py"}
    with ZipFile(out / "connector.zip", "w", ZIP_DEFLATED) as archive:
        for name, path in sources.items():
            archive.writestr(name, path.read_bytes())
    template = root / "deploy/aws/template.yaml"
    (out / "template.yaml").write_bytes(template.read_bytes())
    manifest = {"kind": "aws-mcp-connector-build", "sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sources.items()},
                "zip_sha256": hashlib.sha256((out / "connector.zip").read_bytes()).hexdigest(),
                "deployed": False, "contains_model_weights": False,
                "notes": ["Uses boto3 provided by the Lambda Python runtime.", "Deployment requires your AWS credentials, private code bucket, OAuth callbacks and confirmed account allowances."]}
    (out / "build.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    build(args.out)
    print(args.out / "connector.zip")
