"""Definiciones de consulta y clasificación para datos de referencia Materials Project.

No contiene datos ficticios ni implementaciones alternativas incompatibles.
El pipeline de Kaggle se coordina desde ``analizar_kaggle.py``.
"""

# Estas son las columnas de Materials Project que necesitaremos.
# Guardo los nombres aquí para no repetirlos por todo el programa.
FIELDS = [
    "material_id", "formula_pretty", "energy_above_hull",
    "formation_energy_per_atom", "density", "band_gap",
    "volume", "nsites", "elements", "chemsys", "is_stable",
]
# Clasifico por elementos presentes; es una etiqueta de organización,
# no una prueba de tipo de electrolito.
FAMILIES = {
    "oxidos": "O", "sulfuros": "S", "fluoruros": "F",
    "cloruros": "Cl", "bromuros": "Br", "yoduros": "I",
    "fosfatos_potenciales": "P",
}


# Convierto la lista de elementos en etiquetas sencillas
# que luego facilitan las tablas y la comparación por familias.
def family(elements):
    els = set(elements)
    tags = []
    if "O" in els: tags.append("oxido")
    if "S" in els: tags.append("sulfuro")
    if els & {"F", "Cl", "Br", "I"}: tags.append("haluro")
    if {"P", "O"} <= els: tags.append("fosfato")
    return ";".join(tags) if tags else "otra"
