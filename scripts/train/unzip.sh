find /root/autodl-tmp/LLaVA_Train/ -maxdepth 2 -type f -name "*.zip" -print0 | parallel -0 --bar "sh -c 'dest=\$(echo {} | sed -e \"s|^/root/autodl-tmp/LLaVA_Train|/root/autodl-tmp/datasets/LLaVA_Train|\" -e \"s|\\.zip\$||\"); mkdir -p \"\$dest\"; unzip -qo {} -d \"\$dest\"'"

