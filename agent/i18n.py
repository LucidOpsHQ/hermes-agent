"""Lightweight i18n for Hermes' static user-facing strings (approval prompts, gateway replies, CLI, tips).

Catalogs are flat dotted-key mappings resolved through layers, top first:

1. plugin language packs (``PluginContext.register_locale``; last registered wins) — ``agent.i18n_layers``
2. the user overlay ``$HERMES_HOME/locales/<lang>.yaml`` of the current profile home
3. the bundled ``locales/<lang>.yaml``
4. the same chain for ``en``
5. the bare key

Every layer may be partial. Language resolution: explicit ``lang=`` > ``HERMES_LANGUAGE`` >
``display.language`` > ``en``; any id that some layer supplies is accepted, so a pack-only language
(``pl``) works the moment its plugin loads. ``t()`` is a hot path: one cached merged dict per
``(home, lang)``, invalidated by :func:`reset_language_cache` (which every pack registration calls).
"""

from __future__ import annotations

import logging
import os
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

from agent import i18n_layers
from agent.i18n_languages import language_options

logger = logging.getLogger(__name__)

# Bundled catalogs (compat: tests and the parity check iterate this). ``supported_languages()`` is the
# live set including overlay and pack languages.
SUPPORTED_LANGUAGES: tuple[str, ...] = (
    "en", "zh", "zh-hant", "ja", "de", "es", "fr", "tr", "uk",
    "af", "ko", "it", "ga", "pt", "ru", "hu", "ar",
)
DEFAULT_LANGUAGE = "en"

# Natural aliases so "chinese" / "zh-CN" / "jp" hit the right catalog instead of
# silently falling back to English. Bare "chinese" defaults to Simplified;
# Taiwan/HK/Macau tags route to the distinct Traditional catalog. pt-br shares
# the pt catalog unless a pack supplies a real pt-br one (a supplied id always wins over an alias).
_LANGUAGE_ALIASES: dict[str, str] = {
    "english": "en", "en-us": "en", "en-gb": "en",
    "chinese": "zh", "mandarin": "zh", "zh-cn": "zh", "zh-hans": "zh", "zh-sg": "zh",
    "traditional-chinese": "zh-hant", "traditional_chinese": "zh-hant",
    "zh-tw": "zh-hant", "zh-hk": "zh-hant", "zh-mo": "zh-hant",
    "japanese": "ja", "jp": "ja", "ja-jp": "ja",
    "german": "de", "deutsch": "de", "de-de": "de", "de-at": "de", "de-ch": "de",
    "spanish": "es", "español": "es", "espanol": "es", "es-es": "es", "es-mx": "es", "es-ar": "es",
    "french": "fr", "français": "fr", "france": "fr", "fr-fr": "fr", "fr-be": "fr", "fr-ca": "fr", "fr-ch": "fr",
    "ukrainian": "uk", "ukrainisch": "uk", "українська": "uk", "uk-ua": "uk", "ua": "uk",
    "turkish": "tr", "türkçe": "tr", "tr-tr": "tr",
    "afrikaans": "af", "af-za": "af",
    "korean": "ko", "한국어": "ko", "ko-kr": "ko",
    "italian": "it", "italiano": "it", "it-it": "it", "it-ch": "it",
    "irish": "ga", "gaeilge": "ga", "ga-ie": "ga",
    "portuguese": "pt", "português": "pt", "portugues": "pt",
    "pt-pt": "pt", "pt-br": "pt", "brazilian": "pt", "brasileiro": "pt",
    "russian": "ru", "русский": "ru", "ru-ru": "ru",
    "hungarian": "hu", "magyar": "hu", "hu-hu": "hu",
    "arabic": "ar", "العربية": "ar",
    "ar-sa": "ar", "ar-eg": "ar", "ar-ae": "ar", "ar-ma": "ar", "ar-dz": "ar",
}

# (home, lang) -> merged catalog (packs over overlay over bundled). home -> supported tuple.
_catalog_cache: dict[tuple[str, str], dict[str, str]] = {}
_supported_cache: dict[str, tuple[str, ...]] = {}
_catalog_lock = threading.Lock()


def _locales_dir() -> Path:
    """Locale dir: ``HERMES_BUNDLED_LOCALES`` (sealed packaging, e.g. Nix) if it exists, else ``<repo-root>/locales``.

    The source path is returned even when missing so ``_load_bundled`` can log
    the path it looked at rather than raise.
    """
    override = os.getenv("HERMES_BUNDLED_LOCALES", "").strip()
    if override and Path(override).is_dir():
        return Path(override)
    if override:
        logger.warning(
            "HERMES_BUNDLED_LOCALES points to a non-directory path (%s); "
            "falling back to bundled/source locale resolution", override,
        )
    return Path(__file__).resolve().parent.parent / "locales"


def _current_home() -> str:
    from hermes_constants import get_hermes_home
    return str(get_hermes_home())


def supported_languages(home: str | None = None) -> tuple[str, ...]:
    """Every language some layer supplies for the current profile home: bundled ∪ user overlay ∪ plugin
    packs, ``en`` first then sorted. Cached until :func:`reset_language_cache`."""
    home = home or _current_home()
    with _catalog_lock:
        cached = _supported_cache.get(home)
        if cached is not None:
            return cached
    langs = set(SUPPORTED_LANGUAGES) | i18n_layers.layered_languages(home)
    result = (DEFAULT_LANGUAGE, *sorted(langs - {DEFAULT_LANGUAGE}))
    with _catalog_lock:
        _supported_cache[home] = result
    return result


def resolve_language_id(value: Any, home: str | None = None) -> str | None:
    """Canonical supported id for a user-supplied value (code, alias, regional tag), or ``None`` when no
    layer supplies it — the validation ``hermes config set display.language`` runs."""
    key = i18n_layers.normalize_language_id(value)
    if not key:
        return None
    supported = supported_languages(home)
    if key in supported:
        return key
    alias = _LANGUAGE_ALIASES.get(key)
    if alias in supported:
        return alias
    base = key.split("-", 1)[0]  # strip region suffix
    return base if base in supported else None


def _normalize_lang(value: Any, home: str | None = None) -> str:
    """Map a user-supplied value to a supported code (any layer), else the default."""
    return resolve_language_id(value, home) or DEFAULT_LANGUAGE


def _load_bundled(lang: str) -> dict[str, str]:
    """One bundled locale YAML flattened to dotted keys (empty dict on any failure — never crashes)."""
    path = _locales_dir() / f"{lang}.yaml"
    if not path.is_file():
        logger.debug("i18n catalog missing for %s at %s", lang, path)
        return {}
    try:
        return i18n_layers.parse_locale_file(path)
    except Exception as exc:
        logger.warning("Failed to load i18n catalog %s: %s", path, exc)
        return {}


def _flatten_into(node: Any, prefix: str, out: dict[str, str]) -> None:
    """Flatten a nested mapping into ``out`` (text leaves only)."""
    i18n_layers.flatten(node, prefix, out)


def _load_catalog(lang: str, home: str | None = None) -> dict[str, str]:
    """Merged catalog for ``lang`` in ``home`` (packs > overlay > bundled); cached per (home, lang)."""
    home = home or _current_home()
    key = (home, lang)
    with _catalog_lock:
        cached = _catalog_cache.get(key)
        if cached is not None:
            return cached
    merged = _load_bundled(lang)
    merged.update(i18n_layers.overlay_layer(home, lang))
    merged.update(i18n_layers.pack_layer(lang))
    with _catalog_lock:
        _catalog_cache[key] = merged
    return merged


def surface_catalog(lang: str, surface: str = i18n_layers.CORE_SURFACE) -> dict[str, str]:
    """What ``i18n.catalog`` serves for one surface: packs > overlay > bundled ``locales/<lang>.<surface>.yaml``.

    The core surface omits the bundled layer (the Python renderer already has it); the TUI ships only an
    English catalog in-tree, so its bundled translations live in ``locales/<lang>.tui.yaml`` and ride
    along here."""
    lang = _normalize_lang(lang)
    merged: dict[str, str] = {}
    if surface != i18n_layers.CORE_SURFACE:
        merged.update(_load_bundled(f"{lang}.{surface}"))
    merged.update(i18n_layers.surface_catalog(_current_home(), lang, surface))
    return merged


@lru_cache(maxsize=8)
def _config_language_cached(hermes_home: str) -> str | None:
    """``display.language`` from config.yaml, read once per profile home (``t()`` is a hot path).
    Keyed by home so a multiplexed gateway serving several profiles doesn't freeze the first
    profile's language for every other profile."""
    try:
        from hermes_cli.config import load_config_readonly
        lang = (load_config_readonly().get("display") or {}).get("language")
        return _normalize_lang(lang, hermes_home) if lang else None
    except Exception as exc:
        logger.debug("Could not read display.language from config: %s", exc)
        return None


def _config_language() -> str | None:
    return _config_language_cached(_current_home())


def reset_language_cache() -> None:
    """Invalidate cached language resolution, merged catalogs and every layer view (call after
    ``save_config`` changes ``display.language``, after a pack registers/unregisters, or after editing
    an overlay file)."""
    _config_language_cached.cache_clear()
    with _catalog_lock:
        _catalog_cache.clear()
        _supported_cache.clear()
    i18n_layers.clear_cache()


def _resolve_language(home: str) -> str:
    from agent.secret_scope import UnscopedSecretError, get_secret
    try:
        env_lang = get_secret("HERMES_LANGUAGE")
    except UnscopedSecretError:
        env_lang = os.environ.get("HERMES_LANGUAGE")  # unscoped default-profile path: environ IS its own value
    return _normalize_lang(env_lang, home) if env_lang else _config_language_cached(home) or DEFAULT_LANGUAGE


def get_language() -> str:
    """Resolve the active language using env > config > default order. ``HERMES_LANGUAGE`` is a
    per-profile ``.env`` value, so it is read through the secret scope: under multiplexing a raw
    environ read would impose the default profile's language on every other profile."""
    return _resolve_language(_current_home())


def get_agent_name(default: str = "Hermes") -> str:
    """Display name from the current profile, resolved at render time (never cached).

    Like language, AGENT_NAME uses the native scoped .env reader. Missing profile
    scope under multiplexing remains an error rather than borrowing the launch name.
    """
    from agent.secret_scope import get_secret
    return (get_secret("AGENT_NAME") or "").strip() or default


# Only owned presentation copy: software/support/update references stay Hermes.
# Apply before interpolating arguments so user/session names are never rewritten.
# Upstream moved the gateway/platform copy into the catalog, so agent-facing labels are keyed here
# instead of being f-strings at each call site. ``gateway.update.*``, ``gateway.debug.*``,
# ``gateway.errors.*`` and IRC quit text name the software and deliberately stay Hermes.
_AGENT_NAME_KEYS = frozenset({
    "gateway.help.header", "gateway.status.header",
    "gateway.resume.matrix_blocked_no_origin", "gateway.topic.thread_ready",
    # Command approval heading (card, text fallback, Discord/Feishu/Teams cards).
    "gateway.exec_approval.header", "platform.discord.approval.question",
    # Lifecycle notices.
    "gateway.startup.online", "gateway.startup.handoff_thread_title",
    "gateway.shutdown.notice_shutdown", "gateway.shutdown.notice_restart",
    "gateway.shutdown.cron_interrupted", "gateway.notify.no_home_channel",
    # Busy / pause.
    "gateway.busy.behavior_queue_long", "gateway.busy.behavior_interrupt_long",
    "gateway.pause.not_paused", "gateway.pause.already_paused",
    # Progress task cards.
    "gateway.progress.task_card_title", "platform.slack.task_card.title",
    # Telegram topics.
    "gateway.topic.default_title", "gateway.topic.root_lobby", "gateway.topic.root_new",
    "gateway.topic.new_header", "gateway.topic.help", "gateway.topic.system_topic_intro",
    "gateway.topic.mode_disabled", "gateway.topic.status_how_to",
    "gateway.topic.restored_last_message",
    # Discord (native slash descriptions except /update, threads, clarify prompt).
    "platform.discord.prompt.clarify_title", "platform.discord.command.reset.description",
    "platform.discord.command.status.description", "platform.discord.command.stop.description",
    "platform.discord.command.restart.description", "platform.discord.command.thread.description",
    "platform.discord.command.thread.arg_message", "platform.discord.command.skill.description",
    "platform.discord.thread.seed_message", "platform.discord.thread.default_name",
    "platform.discord.thread.handoff_seed", "platform.discord.thread.auto_create_failed",
    # Other platform labels.
    "platform.slack.handoff.seed", "platform.matrix.handoff.default_name",
    "platform.google_chat.typing.thinking", "platform.email.standalone_subject",
})


def t(key: str, lang: str | None = None, **format_kwargs: Any) -> str:
    """Translate a dotted catalog key to the active (or explicit ``lang``) language.

    ``format_kwargs`` are applied with ``str.format``. Falls back to English,
    then to the bare key; a format failure returns the unformatted string.
    """
    home = _current_home()
    target = _normalize_lang(lang, home) if lang else _resolve_language(home)
    value = _load_catalog(target, home).get(key)
    if value is None and target != DEFAULT_LANGUAGE:
        value = _load_catalog(DEFAULT_LANGUAGE, home).get(key)
    if value is None:
        logger.debug("i18n miss: key=%r lang=%r", key, target)
        value = key
    unformatted = value
    if key in _AGENT_NAME_KEYS:
        value = value.replace("Hermes", "{agent_name}")
        format_kwargs["agent_name"] = get_agent_name()
        unformatted = unformatted.replace("Hermes", format_kwargs["agent_name"])
    if not format_kwargs:
        return value
    try:
        return value.format(**format_kwargs)
    except (KeyError, IndexError, ValueError) as exc:
        logger.warning("i18n format failed for key=%r lang=%r kwargs=%r: %s", key, target, format_kwargs, exc)
        return unformatted


__all__ = [
    "DEFAULT_LANGUAGE",
    "SUPPORTED_LANGUAGES",
    "get_agent_name",
    "get_language",
    "language_options",
    "reset_language_cache",
    "resolve_language_id",
    "supported_languages",
    "surface_catalog",
    "t",
]
