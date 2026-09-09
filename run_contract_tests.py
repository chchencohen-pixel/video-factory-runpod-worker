from test_handler_contract import (
    test_dockerfile_uses_runtime_and_sdpa_safe_dependency_install,
    test_worker_declares_wan_720p_contract,
    test_worker_keeps_the_model_resident_between_jobs,
    test_worker_uses_presigned_urls_and_no_application_secret,
)


def main() -> None:
    test_worker_uses_presigned_urls_and_no_application_secret()
    test_worker_declares_wan_720p_contract()
    test_worker_keeps_the_model_resident_between_jobs()
    test_dockerfile_uses_runtime_and_sdpa_safe_dependency_install()
    print("Worker contract tests passed")


if __name__ == "__main__":
    main()
