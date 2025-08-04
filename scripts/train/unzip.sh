find /root/autodl-tmp/datasets/LLaVA_Train/LLaVA_SFT/DZips -maxdepth 1 -type f -name "*.zip" -print0 | parallel -0 --bar "sh -c 'dest=\$(echo {} | sed -e \"s|^/root/autodl-tmp/datasets/LLaVA_Train/LLaVA_SFT/DZips|/root/autodl-tmp/datasets/LLaVA_Train/LLaVA_SFT|\" -e \"s|\\.zip\$||\"); mkdir -p \"\$dest\"; unzip -qo {} -d \"\$dest\"'"

