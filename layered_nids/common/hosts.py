"""One rule for 'which host does this belong to', shared by every layer so they attribute evidence the same way."""
from __future__ import annotations

import ipaddress


def valid_internal(a: str) -> bool:
    """A real internal unicast address: private, and not unspecified (0.0.0.0), multicast, loopback or reserved (which
    includes 255.255.255.255). MAC addresses and anything that is not an IP are not hosts."""
    try:
        ip = ipaddress.ip_address(a)
    except ValueError:
        return False
    return ip.is_private and not (ip.is_unspecified or ip.is_multicast or ip.is_loopback or ip.is_reserved)


def pick_internal(*addresses: str) -> str:
    """The first valid internal address among `addresses` (in the order given), or '' if there is none."""
    for a in addresses:
        if valid_internal(a):
            return a
    return ""
