variable "DOCKER_REGISTRY_URL" {
    default = ""
}
variable "MUSUBI_VERSION" {
    default = "main"
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
    ]
}

target "base" {
    context = "."
    dockerfile = "dockerfile.base"
    args = {
        MUSUBI_VERSION = "${MUSUBI_VERSION}"
    }
    tags       = ["${DOCKER_REGISTRY_URL}musubi-tuner:${IMAGE_LABEL}"]
    platforms = [ "linux/amd64" ]
    cache-from = ["type=registry,ref=${DOCKER_REGISTRY_URL}musubi-tuner:${IMAGE_LABEL}"]
    cache-to   = ["type=inline"]
}
