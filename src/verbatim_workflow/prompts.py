"""Prompt builders (FR/EN). Pure string logic, no SDK import."""

from __future__ import annotations

from .config import PipelineConfig


def _taxonomy_block(config: PipelineConfig) -> str:
    lines = []
    for cat in config.categories:
        desc = cat.description.strip().replace("\n", " ")
        lines.append(f"- {cat.name}: {desc}" if desc else f"- {cat.name}")
    return "\n".join(lines)


def classification_system_prompt(config: PipelineConfig) -> str:
    taxonomy = _taxonomy_block(config)
    other = config.other_label
    if config.language == "fr":
        multi = (
            "Un verbatim peut relever de PLUSIEURS catégories : renvoie toutes celles qui s'appliquent."
            if config.multilabel
            else "Choisis la SEULE meilleure catégorie."
        )
        return (
            "Tu es un expert en analyse de verbatims clients. "
            "Tu classes chaque verbatim dans une taxonomie fermée.\n\n"
            f"Catégories disponibles :\n{taxonomy}\n"
            f"- {other} : à utiliser UNIQUEMENT si aucune catégorie ci-dessus ne convient.\n\n"
            f"{multi}\n"
            f"N'invente jamais de catégorie hors de cette liste. "
            f"Si le verbatim est vide, ambigu ou hors-sujet, utilise « {other} ». "
            "Donne une justification courte (une phrase) et un niveau de confiance."
        )
    multi = (
        "A verbatim may belong to SEVERAL categories: return all that apply."
        if config.multilabel
        else "Pick the SINGLE best category."
    )
    return (
        "You are an expert at analysing customer verbatims. "
        "You classify each verbatim into a closed taxonomy.\n\n"
        f"Available categories:\n{taxonomy}\n"
        f"- {other}: use ONLY when none of the categories above fits.\n\n"
        f"{multi}\n"
        "Never invent a category outside this list. "
        f"If the verbatim is empty, ambiguous or off-topic, use \"{other}\". "
        "Give a short one-sentence rationale and a confidence level."
    )


def classification_user_prompt(text: str, language: str = "fr") -> str:
    text = (text or "").strip()
    if language == "fr":
        return f"Verbatim à classer :\n\"\"\"\n{text}\n\"\"\""
    return f"Verbatim to classify:\n\"\"\"\n{text}\n\"\"\""


def naming_system_prompt(config: PipelineConfig) -> str:
    known = ", ".join(config.category_names) or "(none)"
    if config.language == "fr":
        return (
            "Tu aides à faire évoluer une taxonomie de verbatims. "
            "On te donne un groupe de verbatims qui n'entraient dans AUCUNE des "
            "catégories existantes et qui partagent un thème commun.\n"
            f"Catégories déjà existantes (à NE PAS reproposer) : {known}.\n"
            "Propose UNE nouvelle catégorie qui décrit ce groupe : un nom court, "
            "une description claire, et un résumé du thème récurrent. "
            "La proposition doit être clairement distincte des catégories existantes."
        )
    return (
        "You help evolve a verbatim taxonomy. You are given a group of verbatims "
        "that fit NONE of the existing categories and share a common theme.\n"
        f"Existing categories (do NOT re-propose these): {known}.\n"
        "Propose ONE new category describing this group: a short name, a clear "
        "description, and a summary of the recurring theme. It must be clearly "
        "distinct from the existing categories."
    )


def naming_user_prompt(samples: list[str], language: str = "fr") -> str:
    bullets = "\n".join(f"- {s.strip()}" for s in samples if s and s.strip())
    if language == "fr":
        return f"Verbatims du groupe :\n{bullets}"
    return f"Verbatims in the group:\n{bullets}"
