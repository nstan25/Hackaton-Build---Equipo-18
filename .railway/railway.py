from railway_sdk import define_railway, project, service, volume

# This repository manages only its own resources in the environment. Other
# repositories export their own partial name.
# See https://docs.railway.com/infrastructure-as-code#multi-repo-projects
PARTIAL = "bcs"


@define_railway
def main(ctx=None):
    # El volumen ya existe en production y dev: se declara solo para que IaC no lo desmonte.
    data = volume("bcs-volume", region="iad", sizeMB=500)
    bcs = service(
        "bcs",
        build={
            "buildEnvironment": "V3",
            "builder": "DOCKERFILE",
            "dockerfilePath": "Dockerfile",
            "watchPatterns": [
                "src/**",
                "config/**",
                "web/**",
                "models/**",
                "train_cnn.py",
                "metrics.py",
                "prep_yolo.py",
                "prep_crops.py",
                "Dockerfile",
                "pyproject.toml",
                "uv.lock",
            ],
        },
        healthcheck="/health",
        healthcheckTimeout=120,
        replicas={"iad": 1},
        deploy={
            "sleepApplication": True,
            "restartPolicyType": "ON_FAILURE",
            "restartPolicyMaxRetries": 5,
        },
        volumeMounts={"/data": data},
    )
    return project("hopeful-cooperation", variables={"managed": False}, resources=[data, bcs])
