find /mnt/LLaVA_Train/ -maxdepth 2 -type f -name "*.zip" -print0 | parallel -0 --bar "sh -c 'dest=\$(echo {} | sed -e \"s|^/mnt/LLaVA_Train|/mnt/datasets/LLaVA_Train|\" -e \"s|\\.zip\$||\"); mkdir -p \"\$dest\"; unzip -qo {} -d \"\$dest\"'"

