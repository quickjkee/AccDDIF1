# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['ftfy', 'ftfy.bad_codecs']

package_data = \
{'': ['*']}

install_requires = \
['wcwidth>=0.2.5']

entry_points = \
{'console_scripts': ['ftfy = ftfy.cli:main']}

setup_kwargs = {
    'name': 'ftfy',
    'version': '6.1.1',
    'description': 'Fixes mojibake and other problems with Unicode, after the fact',
    'long_description': '# ftfy: fixes text for you\n\n[![PyPI package](https://badge.fury.io/py/ftfy.svg)](https://badge.fury.io/py/ftfy)\n[![Docs](https://readthedocs.org/projects/ftfy/badge/?version=latest)](https://ftfy.readthedocs.org/en/latest/)\n\n```python\n>>> print(fix_encoding("(à¸‡\'âŒ£\')à¸‡"))\n(ง\'⌣\')ง\n```\n\nThe full documentation of ftfy is available at [ftfy.readthedocs.org](https://ftfy.readthedocs.org). The documentation covers a lot more than this README, so here are\nsome links into it:\n\n- [Fixing problems and getting explanations](https://ftfy.readthedocs.io/en/latest/explain.html)\n- [Configuring ftfy](https://ftfy.readthedocs.io/en/latest/config.html)\n- [Encodings ftfy can handle](https://ftfy.readthedocs.io/en/latest/encodings.html)\n- [“Fixer” functions](https://ftfy.readthedocs.io/en/latest/fixes.html)\n- [Is ftfy an encoding detector?](https://ftfy.readthedocs.io/en/latest/detect.html)\n- [Heuristics for detecting mojibake](https://ftfy.readthedocs.io/en/latest/heuristic.html)\n- [Support for “bad” encodings](https://ftfy.readthedocs.io/en/latest/bad_encodings.html)\n- [Command-line usage](https://ftfy.readthedocs.io/en/latest/cli.html)\n- [Citing ftfy](https://ftfy.readthedocs.io/en/latest/cite.html)\n\n## Testimonials\n\n- “My life is livable again!”\n  — [@planarrowspace](https://twitter.com/planarrowspace)\n- “A handy piece of magic”\n  — [@simonw](https://twitter.com/simonw)\n- “Saved me a large amount of frustrating dev work”\n  — [@iancal](https://twitter.com/iancal)\n- “ftfy did the right thing right away, with no faffing about. Excellent work, solving a very tricky real-world (whole-world!) problem.”\n  — Brennan Young\n- “I have no idea when I’m gonna need this, but I’m definitely bookmarking it.”\n  — [/u/ocrow](https://reddit.com/u/ocrow)\n- “9.2/10”\n  — [pylint](https://bitbucket.org/logilab/pylint/)\n\n## What it does\n\nHere are some examples (found in the real world) of what ftfy can do:\n\nftfy can fix mojibake (encoding mix-ups), by detecting patterns of characters that were clearly meant to be UTF-8 but were decoded as something else:\n\n    >>> import ftfy\n    >>> ftfy.fix_text(\'âœ” No problems\')\n    \'✔ No problems\'\n\nDoes this sound impossible? It\'s really not. UTF-8 is a well-designed encoding that makes it obvious when it\'s being misused, and a string of mojibake usually contains all the information we need to recover the original string.\n\nftfy can fix multiple layers of mojibake simultaneously:\n\n    >>> ftfy.fix_text(\'The Mona Lisa doesnÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢t have eyebrows.\')\n    "The Mona Lisa doesn\'t have eyebrows."\n\nIt can fix mojibake that has had "curly quotes" applied on top of it, which cannot be consistently decoded until the quotes are uncurled:\n\n    >>> ftfy.fix_text("l’humanitÃ©")\n    "l\'humanité"\n\nftfy can fix mojibake that would have included the character U+A0 (non-breaking space), but the U+A0 was turned into an ASCII space and then combined with another following space:\n\n    >>> ftfy.fix_text(\'Ã\\xa0 perturber la rÃ©flexion\')\n    \'à perturber la réflexion\'\n    >>> ftfy.fix_text(\'Ã perturber la rÃ©flexion\')\n    \'à perturber la réflexion\'\n\nftfy can also decode HTML entities that appear outside of HTML, even in cases where the entity has been incorrectly capitalized:\n\n    >>> # by the HTML 5 standard, only \'P&Eacute;REZ\' is acceptable\n    >>> ftfy.fix_text(\'P&EACUTE;REZ\')\n    \'PÉREZ\'\n  \nThese fixes are not applied in all cases, because ftfy has a strongly-held goal of avoiding false positives -- it should never change correctly-decoded text to something else.\n\nThe following text could be encoded in Windows-1252 and decoded in UTF-8, and it would decode as \'MARQUɅ\'. However, the original text is already sensible, so it is unchanged.\n\n    >>> ftfy.fix_text(\'IL Y MARQUÉ…\')\n    \'IL Y MARQUÉ…\'\n\n## Installing\n\nftfy is a Python 3 package that can be installed using `pip`:\n\n    pip install ftfy\n\n(Or use `pip3 install ftfy` on systems where Python 2 and 3 are both globally\ninstalled and `pip` refers to Python 2.)\n\n### Local development\n\nftfy is developed using `poetry`. Its `setup.py` is vestigial and is not the\nrecommended way to install it.\n\n[Install Poetry](https://python-poetry.org/docs/master/#installing-with-the-official-installer), check out this repository, and run `poetry install` to install ftfy for local development, such as experimenting with the heuristic or running tests.\n\n## Who maintains ftfy?\n\nI\'m Robyn Speer, also known as Elia Robyn Lake. You can find me\n[on GitHub](https://github.com/rspeer) or [Twitter](https://twitter.com/r_speer).\n\n## Citing ftfy\n\nftfy has been used as a crucial data processing step in major NLP research.\n\nIt\'s important to give credit appropriately to everyone whose work you build on\nin research. This includes software, not just high-status contributions such as\nmathematical models. All I ask when you use ftfy for research is that you cite\nit.\n\nftfy has a citable record [on Zenodo](https://zenodo.org/record/2591652).\nA citation of ftfy may look like this:\n\n    Robyn Speer. (2019). ftfy (Version 5.5). Zenodo.\n    http://doi.org/10.5281/zenodo.2591652\n\nIn BibTeX format, the citation is::\n\n    @misc{speer-2019-ftfy,\n      author       = {Robyn Speer},\n      title        = {ftfy},\n      note         = {Version 5.5},\n      year         = 2019,\n      howpublished = {Zenodo},\n      doi          = {10.5281/zenodo.2591652},\n      url          = {https://doi.org/10.5281/zenodo.2591652}\n    }\n',
    'author': 'Robyn Speer',
    'author_email': 'rspeer@arborelia.net',
    'maintainer': None,
    'maintainer_email': None,
    'url': None,
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'entry_points': entry_points,
    'python_requires': '>=3.7,<4',
}


setup(**setup_kwargs)
