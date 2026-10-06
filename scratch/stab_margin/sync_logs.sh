#!/bin/bash
# pull cell train.logs from deckard to ~/stab_margin_2026-10-06/logs/<cell>/train.log
mkdir -p ~/stab_margin_2026-10-06/logs
rsync -a --include='*/' --include='train.log' --exclude='*' jaden@deckard:stab_margin_2026-10-06/runs/ ~/stab_margin_2026-10-06/logs/
