# -*- coding: utf-8 -*-
# @Author: gunjianpan
# @Date:   2019-04-30 13:26:25
# @Last Modified by:   gunjianpan
# @Last Modified time: 2019-04-30 14:37:44

# get sudo auth
sudo -i

# update apt
apt update
apt-get upgrade <<<'Y'

# apt get
apt install git zsh curl
chsh -s /bin/zsh

# install oh-my-zsh
sh -c "$(curl -fsSL https://raw.github.com/robbyrussell/oh-my-zsh/master/tools/install.sh)"

# syntax highlighting
git clone https://github.com/zsh-users/zsh-syntax-highlighting $ZSH_CUSTOM/plugins/zsh-syntax-highlighting
echo "source \$ZSH_CUSTOM/plugins/zsh-syntax-highlighting/zsh-syntax-highlighting.zsh" >>${ZDOTDIR:-$HOME}/.zshrc

# zsh-autosuggestions
git clone git://github.com/zsh-users/zsh-autosuggestions $ZSH_CUSTOM/plugins/zsh-autosuggestions
echo "source \$ZSH_CUSTOM/plugins/zsh-autosuggestions/zsh-autosuggestions.zsh" >>${ZDOTDIR:-$HOME}/.zshrc

# change ~/.zshrc
sed -i 's/plugins=(git)/plugins=(git docker zsh-autosuggestions)/' ${ZDOTDIR:-$HOME}/.zshrc

# soure
source ${ZDOTDIR:-$HOME}/.zshrc

# install fzf & bind default key-binding
git clone --depth 1 https://github.com/junegunn/fzf ${ZDOTDIR:-$HOME}/.fzf
bash ${ZDOTDIR:-$HOME}/.fzf/install <<<'yyy'
source ${ZDOTDIR:-$HOME}/.zshrc

# install fd, url from https://github.com/sharkdp/fd/releases
wget https://github.com/sharkdp/fd/releases/download/v7.2.0/fd_7.2.0_amd64.deb
dpkg -i fd_7.2.0_amd64.deb

# alter filefind to fd
echo "export FZF_DEFAULT_COMMAND='fd --type file'" >>${ZDOTDIR:-$HOME}/.zshrc
echo "export FZF_CTRL_T_COMMAND=\$FZF_DEFAULT_COMMAND" >>${ZDOTDIR:-$HOME}/.zshrc
echo "export FZF_ALT_C_COMMAND='fd -t d . '" >>${ZDOTDIR:-$HOME}/.zshrc

source ~/.zshrc

# Ctrl+R History command; Ctrl+R file catalog
# if you want to DIY key of like 'Atl + C'
# maybe line-num is not 64, but must nearby
sed -i 's/\\ec/^\\/' ${ZDOTDIR:-$HOME}/.fzf/shell/key-bindings.zsh

source ${ZDOTDIR:-$HOME}/.fzf/shell/key-bindings.zsh
