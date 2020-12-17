class topic_vector_dim:
    class __topic_vector_dim:
        def __init__(self, arg):
            self.val = arg
        def __str__(self):
            return repr(self) + self.val
    instance = None
    def __init__(self, arg):
        if not topic_vector_dim.instance:
            topic_vector_dim.instance = topic_vector_dim.__topic_vector_dim(arg)
        else:
            topic_vector_dim.instance.val = arg
    def __getattr__(self, name):
        return getattr(self.instance, name)