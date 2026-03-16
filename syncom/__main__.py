"""Allow ``python -m syncom`` to run the phrase-check CLI without warnings."""
from syncom.phrase_check import main
import sys

sys.exit(main())
