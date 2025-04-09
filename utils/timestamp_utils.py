import time

def parse_timestamp(timestamp_str, fallback=None):
    """Parse a timestamp string into Unix timestamp (seconds since epoch).
    
    Args:
        timestamp_str: String timestamp in either Unix or ISO format
        fallback: Fallback timestamp to use if parsing fails (default: current time)
        
    Returns:
        Float timestamp in seconds, or fallback time if parsing fails
    """
    if fallback is None:
        fallback = time.time()
        
    if not timestamp_str:
        return fallback
        
    try:
        # Try parsing as float first (Unix timestamp)
        return float(timestamp_str)
    except (ValueError, TypeError):
        try:
            # Try parsing as ISO format
            return time.mktime(time.strptime(timestamp_str.split('.')[0], '%Y-%m-%dT%H:%M:%S'))
        except (ValueError, TypeError):
            return fallback  # Fallback to provided time or current time 