class Cache:
    def __init__(self):
        self.cache = {}
    
    def get(self, key):
        return self.cache.get(key, None)
    
    def update(self, **kwargs):
        self.cache.update(kwargs)

class AttnCache:
    def __init__(self):
        self.cache_list = []
    
    def get_length(self):
        return len(self.cache_list)
    
    def get(self, index):
        while len(self.cache_list) <= index:
            self.cache_list.append(Cache())
        return self.cache_list[index]
    
    def update(self, index, **kwargs):
        while len(self.cache_list) <= index:
            self.cache_list.append(Cache())
        self.cache_list[index].update(**kwargs)