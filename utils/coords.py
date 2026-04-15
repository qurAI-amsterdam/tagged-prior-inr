"""
View-name constants for cardiac MRI coordinate spaces.

These constants are kept in a dependency-free module so they can be safely
imported by both canonical/ (data structures) and models/ (registration)
without creating circular imports.
"""

KEY_SAX_VIEW = "sax"
KEY_SAX_SEG_VIEW = "sax_seg"
KEY_4CH_VIEW = "lax4ch"
KEY_4CH_SEG_VIEW = "lax4ch_seg"
KEY_2CH_VIEW = "lax2ch"
KEY_2CH_SEG_VIEW = "lax2ch_seg"
