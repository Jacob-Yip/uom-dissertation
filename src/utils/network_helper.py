import psutil
import os
import torch.distributed as dist

"""
Helper program of networking-related tasks
"""


def get_network_interfaces():
    print("Available network interfaces:")
    for name in psutil.net_if_addrs().keys():
        print(f"- {name}")


def test_gloo_init(interface_name="Wi-Fi"):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["GLOO_SOCKET_IFNAME"] = interface_name

    try:
        dist.init_process_group(
            "gloo", rank=0, world_size=1)

        print(f"SUCCESS: Gloo init with interface '{interface_name}'")

        dist.destroy_process_group()
    except RuntimeError as e:
        print(f"Raw error message: {e}")
        if "unsupported gloo device" in str(e).lower():
            print(
                f"FAILURE: Unsupported gloo device on interface '{interface_name}'")
        else:
            print(f"FAILURE: Other error:\n{e}")
