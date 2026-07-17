variable "DOCKER_REGISTRY_URL" {
    default = "ghcr.io/radiatingreverberations/"
}

variable "MUSUBI_VERSION" {
    default = "main"
}

variable "REFRESH_MUSUBI" {
    default = "0"
}

variable "NVIDIA_BASE_IMAGE" {
    default = "ghcr.io/offloadr/base/nvidia-full:py3.12-torch2.11.0-cuda13.0.3"
}

variable "SSH_HOST_ED25519_KEY_B64" {
    default = ""
}

variable "IMAGE_LABEL" {
    default = "latest"
    validation {
        condition     = IMAGE_LABEL == "latest" || IMAGE_LABEL == "main"
        error_message = "The variable 'IMAGE_LABEL' must be 'latest' or 'main'."
    }
}

group "default" {
    targets = [
        "base",
        "ssh",
    ]
}

target "base" {
    context    = "src"
    dockerfile = "dockerfile.base"
    args = {
        BASE_IMAGE      = "${NVIDIA_BASE_IMAGE}"
        MUSUBI_VERSION  = "${MUSUBI_VERSION}"
        REFRESH_MUSUBI  = "${REFRESH_MUSUBI}"
    }
    tags       = ["${DOCKER_REGISTRY_URL}musubi-tuner:${IMAGE_LABEL}"]
    platforms  = ["linux/amd64"]
    cache-from = ["type=registry,ref=${DOCKER_REGISTRY_URL}musubi-tuner:${IMAGE_LABEL}"]
    cache-to   = ["type=inline"]
}

target "ssh" {
    context    = "src"
    dockerfile = "dockerfile.ssh"
    contexts = {
        musubi-base = "target:base"
    }
    args = {
        MUSUBI_BASE_IMAGE         = "musubi-base"
        SSH_HOST_IDENTITY_DIGEST  = sha256(SSH_HOST_ED25519_KEY_B64)
    }
    secret     = SSH_HOST_ED25519_KEY_B64 != "" ? ["id=SSH_HOST_ED25519_KEY_B64,env=SSH_HOST_ED25519_KEY_B64"] : []
    tags       = ["${DOCKER_REGISTRY_URL}musubi-tuner-ssh:${IMAGE_LABEL}"]
    platforms  = ["linux/amd64"]
    cache-from = ["type=registry,ref=${DOCKER_REGISTRY_URL}musubi-tuner-ssh:${IMAGE_LABEL}"]
    cache-to   = ["type=inline"]
}
