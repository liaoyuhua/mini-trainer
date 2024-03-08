__all__ = ['l1_extensions', 'l2_extensions', 'l3_extensions']

L1EXTENSIONS = ['Checkpoint', 'EarlyStopping', 'Logger']

L2EXTENSIONS = L1EXTENSIONS + ['LrScheduler', 'LossPlot']

L3EXTENSIONS = L2EXTENSIONS + ['Accelerate', 'Profiler']


class Template:
    def __init__(self, level: int = 1):
        self.level = level

        if level == 1:
            self.ext = L1EXTENSIONS
        elif level == 2:
            self.ext = L2EXTENSIONS
        elif level == 3:
            self.ext = L3EXTENSIONS

    def __repr__(self):
        return f"**Level {self.level} Template** Following extensions should be implemented: {', '.join(self.ext)}"


l1_extensions = Template(1)
l2_extensions = Template(2)
l3_extensions = Template(3)
