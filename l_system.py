from abc import ABCMeta
import numpy as np
import holoviews as hv


class L_Grammar(object):
    """
    Evolution grammar for L-System
    """
    __metaclass__ = ABCMeta

    def __init__(self, initial, rules):
        self.initial = initial
        self.rules = rules

    def expand(self, iterations=1):
        expansion = self.initial
        for i in range(iterations):
            expansion = "".join([self.rules.get(ch, ch) for ch in expansion])
        return expansion


class SimpleAgent(object):
    __metaclass__ = ABCMeta

    def __init__(self, x0=0, y0=0, phi0=0):
        """
        A Turtle graphics Agent
          (x0, y0): Initial position
          phi0: Initial angle
        """

        self.x, self.y = x0, y0
        self.phi = phi0
        self._trace = [(self.x, self.y)]
        # Stateful agent
        self._state = []
        self._traces = []

    @property
    def trace(self):
        return self._traces + [self._trace]

    def forward(self, r=1.0):
        """
        Move the turtle by r unit in the current direction
        """
        assert r >= 0, "Distance must be non-negative"
        self.x += r*np.cos(2*np.pi*self.phi/360.0)
        self.y += r*np.sin(2*np.pi*self.phi/360.0)
        self._trace.append((self.x, self.y))

    def backward(self, r=1.0):
        """
        Move the turte backward, the direction remains unchanged
        """
        self.forward(-1.0*r)

    def rotate(self, angle):
        """
        Rotate the turtle by angle degree, the turtle stays at the same spot
        """
        self.phi += angle

    def push(self):
        self._state.append((self.x, self.y, self.phi))
        self._traces.append(self._trace)

    def pop(self):
        self.x, self.y, self.phi = self._state.pop()
        self._traces.append(self._trace)
        self._trace = [(self.x, self.y)]

    def noop(self):
        """
        Do nothing
        """
        pass

    def pad_extents(self, path):
        """
            Add 5% padding around the path
        """
        minx, maxx = path.range('x')
        miny, maxy = path.range('y')
        xpadding = ((maxx-minx) * 0.1)/2
        ypadding = ((maxy-miny) * 0.1)/2
        path.extents = (minx-xpadding, miny-ypadding, maxx+xpadding, maxy+ypadding)
        return path

    @property
    def path(self, pad_extends=True):
        if pad_extends:
            return self.pad_extents(hv.Path(self.trace))
        else:
            return hv.Path(self.trace)


class L_Agent(SimpleAgent):
    """
    An agent that follows rules.
    """
    default_rules = {'F': lambda t, d, a: t.forward(d),
                     'B': lambda t, d, a: t.back(d),
                     '+': lambda t, d, a: t.rotate(-a),
                     '-': lambda t, d, a: t.rotate(a),
                     'X': lambda t, d, a: t.noop(),
                     'Y': lambda t, d, a: t.noop(),
                     '[': lambda t, d, a: t.push(),
                     ']': lambda t, d, a: t.pop()}

    def __init__(self, x0=0, y0=0, phi0=0,
                 rules=default_rules,
                 grammar=None,
                 step=5, phi=60,
                 iterations=1):
        assert grammar is not None, "Grammar must be provided"
        SimpleAgent.__init__(self, x0, y0, phi0)
        self.step = step
        self.dphi = phi
        self.rules = rules
        self.iterations = iterations
        self.grammar = grammar
        self._process()

    def _process(self):
        seq = self.grammar.expand(self.iterations)
        for i in seq:
            self.rules[i](self, self.step, self.dphi)
