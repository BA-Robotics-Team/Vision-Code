import depthai as dai

devices = dai.Device.getAllAvailableDevices()

if not devices:
    print("❌ No DepthAI devices found.")
else:
    print(f"✅ Found {len(devices)} device(s):")
    for i, d in enumerate(devices):
        print(f"[{i}]")
        print(f"  MxID : {d.getMxId()}")
        print(f"  Name : {d.name}")
        print(f"  State: {d.state}")
        print(f"  Protocol: {d.protocol}")
        print(f"  IP   : {d.getIp()} (if PoE)")
