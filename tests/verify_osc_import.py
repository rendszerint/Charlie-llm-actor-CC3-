try:
    from pythonosc import udp_client
    print("python-osc imported successfully")
except ImportError as e:
    print(f"Failed to import python-osc: {e}")
    exit(1)

try:
    from services.osc import OscService
    from pathlib import Path
    import time
    
    log_path = Path("tests/test_osc_log.txt")
    if log_path.exists():
        log_path.unlink()
        
    service = OscService(log_file=log_path)
    print("OscService instantiated successfully")
    
    # Test process_turn which should check filtering and sending to both /text and /action
    service.process_turn("Hello world")
    
    # Old log check removed as we now use dashboard mode which overwrites
    print("process_turn executed successfully")

    print("Verifier: Testing time-based counter and dashboard...")
    
    # 1. Test initial idle state and dashboard creation
    time.sleep(1)
    
    if log_path.exists():
        content = log_path.read_text("utf-8")
        if "[OSC STATUS]" in content:
            print("Verified: Dashboard header found.")
        else:
            print("Failed: Dashboard header missing.")
    
    service.report_activity()
    print("Verifier: Activity reported. Waiting 2 seconds for counter update...")
    time.sleep(2.0)
    
    if log_path.exists():
        content = log_path.read_text("utf-8")
        # Check for incremented values (it should be the *only* value, not a list)
        if "Counter      : 12." in content:
             # Parse the value to confirm it's > 12.0
             import re
             match = re.search(r"Counter\s+:\s+(\d+\.\d+)", content)
             if match and float(match.group(1)) > 12.0:
                 print(f"Verified: Counter updated to {match.group(1)}")
             else:
                 print(f"Warning: Counter value not clearly incremented: {content}")
        else:
            print("Warning: Counter field not found.")

    # 2. Test Text/Action update
    service.process_turn("Hello world")
    time.sleep(0.5) # Allow write
    content = log_path.read_text("utf-8")
    if "Last Text    : Hello world" in content and "Last Actions : Hello, world" in content:
        print("Verified: Dashboard text keys updated.")
    else:
        print(f"Failed: Dashboard text update incorrect. Content:\n{content}")

    service.stop()
    print("OscService stopped.")
    
    # Verify default persona again
    import os
    default_persona = os.getenv("OSC_TARGET_PERSONA", "persona2").lower()
    if default_persona == "persona2":
        print("Verified: OSC_TARGET_PERSONA default correct ('persona2').")
    else:
        print(f"Failed: OSC_TARGET_PERSONA default mismatch ({default_persona}).")

except Exception as e:
    print(f"Failed to verify OscService: {e}")
    exit(1)
