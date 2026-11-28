from manim import *
import numpy as np

import json, os

PX = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "steve.json")))
U = 0.13

config.background_color = "#000000"
config.frame_height = 8
config.frame_width = 16


class Me(VGroup):
    """Pixel Arnav, who narrates the way a Pi creature would."""

    def __init__(self, at):
        super().__init__()
        def px(x, y, col):
            return Square(U * 1.06, stroke_width=0, fill_color=col, fill_opacity=1).move_to([(x - 7.5) * U, (15.5 - y) * U, 0])
        is_arm = lambda x, y: x >= 12 and 8 <= y < 20
        self.body = VGroup(*[px(x, y, c) for x, y, c in PX if not is_arm(x, y)])
        self.arm = VGroup(*[px(x, y, c) for x, y, c in PX if is_arm(x, y)])
        self.lids = VGroup(*[px(x, 5, "#c98f6b") for x in (5, 6, 9, 10)]).set_opacity(0)
        self.add(self.body, self.arm, self.lids)
        self.shift(at)

    def shoulder(self):
        return self.arm.get_corner(UL) + RIGHT * 0.5 * U + DOWN * U


def bubble(content, me):
    """A speech bubble above his head, holding whatever he is saying."""
    top = me.body.get_top()
    box = RoundedRectangle(width=content.width + 0.8, height=content.height + 0.65, corner_radius=0.25,
                           stroke_width=0, fill_color=WHITE, fill_opacity=1)
    box.move_to(top + UP * (0.5 + box.height / 2))
    content.move_to(box)
    base = box.get_bottom() + UP * 0.02
    tail = Polygon(top + UP * 0.1, base + LEFT * 0.25, base + RIGHT * 0.25,
                   stroke_width=0, fill_color=WHITE, fill_opacity=1)
    return VGroup(tail, box), content


class Intro(Scene):
    def blink(self, me):
        me.lids.set_opacity(1)
        self.wait(0.15)
        me.lids.set_opacity(0)

    def wave(self, me):
        pivot = me.shoulder()
        for a, d in [(2.5, 0.35), (-0.45, 0.18), (0.45, 0.18), (-0.45, 0.18), (0.45, 0.18), (-2.5, 0.35)]:
            self.play(Rotate(me.arm, a, about_point=pivot), run_time=d)

    def say(self, me, text, hold=1.0, size=48):
        shell, words = bubble(Tex(text, font_size=size, color=BLACK), me)
        self.play(GrowFromPoint(shell, me.body.get_top()), run_time=0.35)
        self.play(Write(words), run_time=min(1.3, 0.3 + 0.035 * len(text)))
        if hold:
            self.wait(hold)
        return VGroup(shell, words)

    def hush(self, b, me):
        self.play(FadeOut(b, target_position=me.body.get_top(), scale=0.3), run_time=0.25)

    def hop(self, me, to):
        self.play(me.animate.move_to(to), run_time=0.8, path_arc=-1.6 if to[0] > me.get_x() else 1.6)

    def doodles(self):
        """AI systems, equations and stray bits, kept to the sides."""
        def at(x, y):
            return np.array([x, y, 0])

        def net(c):
            pts = [[c + at((i - 1.5) * 0.5, (j - (n - 1) / 2) * 0.3) for j in range(n)] for i, n in enumerate([3, 4, 4, 2])]
            edges = VGroup(*[Line(a, b, stroke_width=1, color=GREY) for l, r in zip(pts, pts[1:]) for a in l for b in r])
            nodes = VGroup(*[Circle(0.07, color=BLUE, stroke_width=2.5, fill_color=BLACK, fill_opacity=1).move_to(p) for l in pts for p in l])
            return VGroup(edges, nodes)

        def box(label, c, col, w=1.7):
            r = RoundedRectangle(width=w, height=0.38, corner_radius=0.08, color=col, stroke_width=2.5).move_to(c)
            return VGroup(r, Tex(label, font_size=22, color=col).move_to(c))

        def arrow(a, b, col=GREY):
            return Arrow(a, b, buff=0.02, color=col, stroke_width=2.5, max_tip_length_to_length_ratio=0.3)

        # a transformer block
        t = VGroup(*[box(n, at(6.6, y), col) for n, y, col in
                     (("Self-Attention", 3.35, TEAL), ("Add \\& Norm", 2.75, GREY_B), ("MLP", 2.15, BLUE), ("Softmax", 1.55, YELLOW))])
        t.add(*[arrow(at(6.6, y - 0.19), at(6.6, y - 0.41)) for y in (3.35, 2.75, 2.15)])

        # an agent loop
        llm, tool, mem = at(6.6, -1.7), at(7.25, -3.0), at(5.95, -3.0)
        agent = VGroup(box("LLM", llm, YELLOW, 0.9), box("Tools", tool, TEAL, 0.95), box("Memory", mem, MAROON_B, 1.05),
                       arrow(llm + at(0.25, -0.2), tool + at(0, 0.2)), arrow(tool + at(-0.5, 0), mem + at(0.55, 0)),
                       arrow(mem + at(0, 0.2), llm + at(-0.25, -0.2)))

        def eq(tex, x, y, col, size=26):
            return MathTex(tex, font_size=size, color=col).move_to(at(x, y))

        equations = [
            eq(r"\mathrm{softmax}\!\left(\tfrac{QK^{\top}}{\sqrt{d_k}}\right)\!V", -6.5, 1.55, TEAL_A),
            eq(r"\theta_{t+1}=\theta_t-\eta\nabla\mathcal{L}", -6.55, 0.55, YELLOW_A),
            eq(r"\sigma(z)=\frac{1}{1+e^{-z}}", -6.55, -0.4, BLUE_A),
            eq(r"\mathcal{L}=-\sum_i y_i\log\hat{y}_i", 6.6, 0.55, MAROON_A),
            eq(r"P(\theta\mid D)\propto P(D\mid\theta)P(\theta)", 6.55, -0.5, GREY_B, 23),
        ]

        rng = np.random.default_rng(7)
        bits = VGroup()
        for x, y in [(-7.7, 3.7), (-5.2, 3.5), (-5.3, 2.1), (-7.6, 2.0), (-5.1, 0.0), (-7.7, -0.1), (-5.2, -1.1), (-7.5, -1.0),
                     (-4.4, 3.7), (-3.0, 3.6), (3.2, 3.7), (4.6, 3.4), (5.3, 2.5), (7.8, 1.0), (5.3, 1.1), (5.2, -0.0),
                     (7.8, -1.2), (5.4, -1.2), (4.8, -2.4), (4.4, -3.4), (-4.6, -3.5), (-2.5, -3.6), (2.4, -3.55),
                     (-5.4, 3.0), (7.8, -2.2), (5.2, 3.8)]:
            word = "".join(rng.choice(["0", "1"], size=rng.integers(1, 5)))
            col = [BLUE, TEAL, YELLOW, MAROON_B, GREEN][rng.integers(5)]
            bits.add(Tex(r"\texttt{%s}" % word, font_size=24, color=col).move_to(at(x, y)).set_opacity(0.55))

        return VGroup(net(at(-6.5, 2.75)), t, agent, *equations, bits)

    def confetti(self, n=110, seed=1):
        """A burst of 0s and 1s that pops up and rains down behind the text."""
        rng = np.random.default_rng(seed)
        cols = [BLUE, TEAL, YELLOW, MAROON_B, GREEN, WHITE]
        proto = {d: Tex(r"\texttt{%s}" % d, font_size=34) for d in "01"}
        bits, motion = VGroup(), []
        for _ in range(n):
            m = proto[str(rng.integers(2))].copy().set_color(cols[rng.integers(len(cols))]).rotate(rng.uniform(-1, 1))
            side = rng.choice([-1, 1])
            motion.append((np.array([side * 8.2, rng.uniform(-3.5, 0), 0]),
                           np.array([-side * rng.uniform(1, 8), rng.uniform(4, 11), 0]), rng.uniform(-5, 5)))
            m.move_to(motion[-1][0])
            bits.add(m)
        bits.set_z_index(-1)
        last = [0.0]

        def fall(group, a):
            T = 2.2 * a
            for m, (p0, v, spin) in zip(group, motion):
                m.move_to(p0 + v * T + np.array([0, -4.5 * T * T, 0]))
                m.rotate(spin * (T - last[0]))
                m.set_opacity(min(1, 3 * (1 - a)))
            last[0] = T

        return UpdateFromAlphaFunc(bits, fall, run_time=2.2, rate_func=linear, remover=True)

    def construct(self):
        me = Me(np.array([0, -1.76, 0]))
        art = self.doodles()

        # 1. welcome
        self.play(FadeIn(me, shift=UP), LaggedStart(*[FadeIn(a, scale=0.8) for a in art], lag_ratio=0.12), run_time=1.6)
        b = self.say(me, "hi, I'm Arnav", hold=0.2)
        self.wave(me)
        self.blink(me)
        self.hush(b, me)

        b = self.say(me, "Welcome to my GitHub", hold=0, size=56)
        self.play(self.confetti())
        self.hush(b, me)

        b = self.say(me, "You can find repos of my code here.", hold=1.4)
        self.blink(me)
        self.hush(b, me)

        b = self.say(me, "Thanks!", hold=0.9, size=60)
        self.hush(b, me)

        # 2. how to connect
        b = self.say(me, "Here's how to connect with me", hold=1.0)
        self.blink(me)
        self.hush(b, me)

        rows = VGroup()
        for k, (name, where, col) in enumerate([
            ("Website", "arnavdhiman.com", BLUE_E),
            ("Email", "arnavd371@gmail.com", TEAL_E),
            ("LinkedIn", "linkedin.com/in/arnavd371", DARK_BLUE),
            ("GitHub", "github.com/arnavd371", GREEN_E),
            ("X", "x.com/arnavvvdhiman", MAROON_D),
        ]):
            y = -0.5 * k
            label = Tex(r"\textbf{%s}" % name, font_size=38, color=col).move_to([0, y, 0], RIGHT)
            value = Tex(where, font_size=38, color=BLACK).move_to([0.35, y, 0], LEFT)
            rows.add(VGroup(label, value))
        shell, rows = bubble(rows, me)
        self.play(GrowFromPoint(shell, me.body.get_top()), run_time=0.35)
        for row in rows:
            self.play(Write(row), run_time=0.7)
        self.blink(me)
        self.wait(4.0)
        self.hush(VGroup(shell, rows), me)

        b = self.say(me, "come say hi", hold=0, size=56)
        self.play(self.confetti(130, seed=3))
        self.wave(me)
        self.blink(me)
        self.wait(1.0)
        self.play(FadeOut(b, me, art), run_time=0.8)
