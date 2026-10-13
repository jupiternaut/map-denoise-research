"""Fixed current representation: A9 -> A10D -> A11DC, all computed cold."""
from methods import prepare_method, decode

def prepare(observation,budget_seconds):
    return prepare_method('accumulated',observation,budget_seconds)
