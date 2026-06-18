"""Wan Director feature modules — local re-implementations of every
Kijai feature so we are functionally independent if kijai disappears
tomorrow. The adapter (``..._kijai_adapter``) routes to kijai's
implementation when available + signature-compatible, otherwise to
these local copies.
"""
