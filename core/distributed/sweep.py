# Calculate number of TPUs needed for sweep.

# Ensure TPUs are up and unused.

from collections import Counter
import socket
import time

import ray

ray.init()

@ray.remote
def f():
    time.sleep(0.001)
    # Return IP address.
    return socket.gethostbyname(socket.gethostname())

object_ids = [f.remote() for _ in range(10000)]
ip_addresses = ray.get(object_ids)
print(Counter(ip_addresses))
