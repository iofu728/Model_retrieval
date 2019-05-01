#!/bin/bash
# @Author: gunjianpan
# @Date:   2019-04-30 13:26:25
# @Last Modified time: 2019-05-01 02:03:39
# A zsh deploy shell for ubuntu.
# In this shell, will install zsh, oh-my-zsh, zsh-syntax-highlighting, zsh-autosuggestions, fzf

set -e

# some constant params
FD_VERSION=7.3.0
ZSH_HL=zsh-syntax-highlighting
ZSH_AS=zsh-autosuggestions
ZSH_CUSTOM=${ZSH}/custom
ZSH_P=${ZSH_CUSTOM}/plugins/
ZSH_HL_P=${ZSH_P}${ZSH_HL}
ZSH_AS_P=${ZSH_P}${ZSH_AS}
ZSHRC=${ZDOTDIR:-$HOME}/.zshrc
FZF=${ZDOTDIR:-$HOME}/.fzf
FD_P=fd_${FD_VERSION}_amd64.deb
FD_URL=https://github.com/sharkdp/fd/releases/download/v${FD_VERSION}/${FD_P}

VIM_P=${ZDOTDIR:-$HOME}/.vim_runtime
VIM_URL='https://github.com/amix/vimrc'
VIMRC=${ZDOTDIR:-$HOME}/.vimrc

BASH_SHELL='bash zsh_linux.sh'
SOURCE_SH='source ${ZDOTDIR:-$HOME}/.zshrc'
OH_MY_ZSH_URL='https://raw.github.com/robbyrussell/oh-my-zsh/master/tools/install.sh'
GITHUB='https://github.com/iofu728/zsh.sh'
ZSH_USER_URL='https://github.com/zsh-users/'
ZSH_HL_URL=${ZSH_USER_URL}${ZSH_HL}
ZSH_AS_URL=${ZSH_USER_URL}${ZSH_AS}
SIGN_1='#-#-#-#-#-#-#-#-#-#'
SIGN_2='---__---'
SIGN_3='**************'
INS='Instaling'
DOW='Downloading'

DISTRIBUTION=$(lsb_release -a 2>/dev/null | grep -n 'Distributor ID:.*' | awk '{print $3}' 2>/dev/null)
if [ -z $DISTRIBUTION ]; then
    if [ ! -z $(which yum 2>/dev/null) ]; then
        DISTRIBUTION=CentOS
    elif [ ! -z "$(which apt 2>/dev/null | sed -n '/\/apt/p')" ]; then
        DISTRIBUTION=Ubuntu
    fi
fi

# echo color
RED='\033[1;91m'
GREEN='\033[1;92m'
YELLOW='\033[1;93m'
BLUE='\033[1;94m'
CYAN='\033[1;96m'
NC='\033[0m'

echo_color() {
    case ${1} in
    red)
        echo -e "${RED} ${2} ${NC}"
        ;;
    green)
        echo -e "${GREEN} ${2} ${NC}"
        ;;
    yellow)
        echo -e "${YELLOW} ${2} ${NC}"
        ;;
    blue)
        echo -e "${BLUE} ${2} ${NC}"
        ;;
    cyan)
        echo -e "${CYAN} ${2} ${NC}"
        ;;
    esac
}

check_install() {
    case $DISTRIBUTION in
    Ubuntu)
        if [ -z "$(which ${1} | sed -n '/\/'${1}'/p')" ]; then
            echo_color green "${SIGN_1} ${INS} ${1} ${SIGN_1}"
            apt-get install ${1} -y
        fi
        ;;
    CentOS)
        if [ -z $(which ${1} 2>/dev/null) ]; then
            echo_color green "${SIGN_1} ${INS} ${1} ${SIGN_1}"
            yum install ${1} -y
        fi
        ;;
    *)
        echo_color red "Sorry, this .sh does not support your Linux Distribution ${DISTRIBUTION}. Please open one issue in ${GITHUB} "
        exit 2
        ;;
    esac
}

update_list() {
    case $DISTRIBUTION in
    Ubuntu)
        apt-get update -y
        ;;
    CentOS)
        yum update -y
        ;;
    *)
        echo_color red "Sorry, this .sh does not support your Linux Distribution ${DISTRIBUTION}. Please open one issue in ${GITHUB} "
        exit 1
        ;;
    esac
}

if [ -z "$(ls -a ${ZDOTDIR:-$HOME} | sed -n '/\.oh-my-zsh/p')" ]; then
    update_list
    check_install zsh
    check_install curl
    check_install git
    check_install dpkg
    chsh -s $(which zsh)

    echo_color yellow "${SIGN_1} ${INS} oh-my-zsh ${SIGN_1}"
    echo_color red "${SIGN_3} After Install you should ·${BASH_SH} && ${SOURCE_SH}· Again ${SIGN_3}"
    sh -c "$(curl -fsSL ${OH_MY_ZSH_URL})"
else
    echo_color green "ZSH_CUSTOM: ${ZSH_CUSTOM}"

    # zsh syntax highlighting
    if [ -z "$(ls ${ZSH_P} | sed -n '/'${ZSH_HL}'/p')" ]; then
        echo_color yellow "${SIGN_2} ${DOW} ${ZSH_HL} ${SIGN_2}"
        git clone ${ZSH_HL_URL} ${ZSH_HL_P}
        echo "source \$ZSH_CUSTOM/plugins/${ZSH_HL}/${ZSH_HL}.zsh" >>${ZSHRC}
    fi

    # zsh-autosuggestions
    if [ -z "$(ls ${ZSH_P} | sed -n '/'${ZSH_AS}'/p')" ]; then
        echo_color yellow "${SIGN_2} ${DOW} ${ZSH_AS} ${SIGN_2}"
        git clone ${ZSH_AS_URL} ${ZSH_AS_P}
        echo "source \$ZSH_CUSTOM/plugins/${ZSH_AS}/${ZSH_AS}.zsh" >>${ZSHRC}
    fi

    # change ~/.zshrc
    sed -i 's/plugins=(git)/plugins=(git docker zsh-autosuggestions)/' ${ZSHRC}

    # install fzf & bind default key-binding
    if [ -z "$(ls -a ${ZDOTDIR:-$HOME} | sed -n '/\.fzf/p')" ]; then
        echo_color yellow "${SIGN_2} ${DOW} fzf ${SIGN_2}"
        git clone --depth 1 https://github.com/junegunn/fzf ${FZF}
        echo_color yellow "${SIGN_2} Installing fzf ${SIGN_2}"
        bash ${FZF}/install <<<'yyy'

        # install fd, url from https://github.com/sharkdp/fd/releases
        echo_color yellow "${SIGN_2} ${DOW} fd ${SIGN_2}"
        wget ${FD_URL}
        dpkg -i ${FD_P}

        # alter filefind to fd
        echo "export FZF_DEFAULT_COMMAND='fd --type file'" >>${ZSHRC}
        echo "export FZF_CTRL_T_COMMAND=\$FZF_DEFAULT_COMMAND" >>${ZSHRC}
        echo "export FZF_ALT_C_COMMAND='fd -t d . '" >>${ZSHRC}

        # Ctrl+R History command; Ctrl+R file catalog
        # if you want to DIY key of like 'Atl + C'
        # maybe line-num is not 64, but must nearby
        sed -i 's/\\ec/^\\/' ${FZF}/shell/key-bindings.zsh
    fi

    # vimrc
    if [ -z "$(ls -a ${ZDOTDIR:-$HOME} | sed -n '/\.vim_runtime/p')" ]; then
        git clone --depth=1 ${VIM_URL} ${VIM_P}
        sh ${VIM_P}/install_awesome_vimrc.sh

        echo -e 'set runtimepath+=~/.vim_runtime÷\nset nocompatible
set nu!\nset history=1000\nset autoindent\nset cindent\nset smartindent\nset tabstop=4\nset ai!\nset showmatch\nset guioptions-=T
set vb t_vb=\nset ruler\nset incsearch\n\nsource ~/.vim_runtime/vimrcs/basic.vim\nsource ~/.vim_runtime/vimrcs/filetypes.vim
source ~/.vim_runtime/vimrcs/plugins_config.vim\nsource ~/.vim_runtime/vimrcs/extended.vim' >>${VIMRC}

    fi

    echo_color red "Warning: If you only execute ·${BASH_SH}·. You need ·${SOURCE_SH}· After running this shell."
    echo_color blue 'Zsh deploy finish. Now you can enjoy it💂'
    echo_color yellow "More Info Can Find in https://wyydsb.xin/other/terminal.html & ${GITHUB} 😶"
fi
