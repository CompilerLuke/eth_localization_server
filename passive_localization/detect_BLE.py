import argparse
import asyncio

from bleak import BleakScanner, BleakClient

ADDRESS_PREFIX = "00:FA:B6"
UUID = "58793564" + "459c" + "548d" + "bfcc" + "367ffd4fcd70" ## same for all beacons

## ref to ibeacon format 
## https://stackoverflow.com/questions/18906988/what-is-the-ibeacon-bluetooth-profile/19040616#19040616
def parse_ibeacon_data(hex_data):
    uuid = hex_data[4:36]
    major = int(hex_data[36:40], 16)
    minor = int(hex_data[40:44], 16)
    tx = int(hex_data[44:], 16)
    return uuid, major, minor, tx

def parse_advertisement(a):
    if a.manufacturer_data:
        if 76 in a.manufacturer_data: ## means apple
            s = a.manufacturer_data[76].hex()
            if len(s) == 46: ## correct iBeacon format
                uuid, major, minor, tx = parse_ibeacon_data(s)
                print(uuid)
                if uuid == UUID:
                    return uuid, major, minor, tx
    return None


async def main(args: argparse.Namespace):
    print("scanning for 10 seconds, please wait...")

    devices = await BleakScanner.discover(
        timeout=10,
        return_adv=True, cb=dict(use_bdaddr=args.macos_use_bdaddr)
    )

    for d, a in devices.values():
        if d.address.startswith(ADDRESS_PREFIX):
            print()
            print(d.address)
            print(d.name)
            print("-" * len(str(d)))

            res = parse_advertisement(a)
            if res:
                uuid, major, minor, tx = res
                print(minor)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--macos-use-bdaddr",
        action="store_true",
        help="when true use Bluetooth address instead of UUID on macOS",
    )
    args = parser.parse_args()
    asyncio.run(main(args))
