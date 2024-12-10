# memory.py
# 
# Memory management

import platform

def get_memory_available_gb():
    '''
    Get the available system memory in gigabytes. Only works on linux platforms.

    Note:
        The results of this function are equivalent to the terminal commands:
        * "grep MemAvailable /proc/meminfo" -> available memory
        * "grep MemTotal /proc/meminfo" -> total memory

    Returns:
        int: number of gigabytes of available system memory
    '''
    if platform.system() != "Linux":
        print('Only works on linux!')
        return
    with open('/proc/meminfo', 'r') as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) == 'MemAvailable:':
                free_memory = int(sline[1])
                break
    return int(free_memory/1e6) # convert KB to GB

def set_memory_limit_gb(size_gb):
    '''
    Set a memory resource limit in gigabytes. Only works on linux platforms.

    Note: 
        This function sets a soft limit, not a hard limit. The soft limit is a value upon 
        which the operating system will restrict memory usage by the process (python, 
        in this case). A true upper bound on the memory values can be defined by the hard 
        limit. However, although the hard limit can be lowered, it can never be raised by 
        user processes (even if the process lowered itself) and is controlled by a 
        system-wide parameter set by the system administrator. Nevertheless, the soft
        limit should serve to raise a `MemoryError` whenever python exceeds the setting.

    Args:
        size_gb (int): upper limit of memory that will be made available to python in gigabytes
    '''
    if platform.system() != "Linux":
        print('Only works on linux!')
        return
    import resource
    maxsize = int(1e9*size_gb)
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (maxsize, hard))

def get_memory_limit_gb():
    '''
    Get the memory resource limit in gigabytes. Only works on linux platforms.

    Returns:
        int or None: upper limit of memory available to python in gigabytes
    '''
    if platform.system() != "Linux":
        print('Only works on linux!')
        return
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    if soft == resource.RLIM_INFINITY:
        return
    return int(soft/1e9)
    
def release_memory_limit():
    '''
    Unset any memory resource limit that may have been applied. Only works on 
    linux platforms.
    '''
    if platform.system() != "Linux":
        print('Only works on linux!')
        return
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (resource.RLIM_INFINITY, hard))
