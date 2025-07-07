# Lab 3
from sympy import symbols, diff

# Forward pass values
x = 2
w = -2
b = 8
y = 1

# Forward pass
c = w * x
a = c + b
d = a - y
J = d**2 / 2

print("Forward Pass:")
print(f"c = {c}, a = {a}, d = {d}, J = {J}")

# Symbolic variables
sx, sw, sb, sy = symbols('x w b y')
sc, sa, sd = symbols('c a d')

# Symbolic expressions
sc_expr = sw * sx
sa_expr = sc + sb
sd_expr = sa - sy
sJ_expr = sd**2 / 2

# Backward pass: compute gradients
dJ_dd = diff(sJ_expr, sd)
dd_da = diff(sd_expr, sa)
dJ_da = dJ_dd * dd_da

da_dc = diff(sa_expr, sc)
da_db = diff(sa_expr, sb)
dJ_dc = dJ_da * da_dc
dJ_db = dJ_da * da_db

dc_dw = diff(sc_expr, sw)
dJ_dw = dJ_dc * dc_dw

# Substitute values into expressions
subs = [(sd, d), (sa, a), (sc, c), (sx, x)]
dJ_dw_val = dJ_dw.subs(subs)
dJ_db_val = dJ_db.subs(subs)

print("\nBackpropagation:")
print(f"dJ/dd = {dJ_dd}, dd/da = {dd_da}")
print(f"dJ/da = {dJ_da}")
print(f"da/dc = {da_dc}, da/db = {da_db}")
print(f"dJ/dc = {dJ_dc}, dJ/db = {dJ_db}")
print(f"dc/dw = {dc_dw}")
print(f"dJ/dw = {dJ_dw}")

print("\nFinal Derivatives with values:")
print(f"dJ/dw = {dJ_dw_val}, dJ/db = {dJ_db_val}")
