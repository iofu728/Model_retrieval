set runtimepath+=~/.vim_runtime
set nocompatible              " 去掉有关vi一致性模式，避免以前版本的bug和局限    
set nu!                        " 显示行号
set history=1000               " 记录历史的行数
set autoindent                 " vim使用自动对齐，也就是把当前行的对齐格式应用到下一行(自动缩进）
set cindent                    " （cindent是特别针对 C语言语法自动缩进）
set smartindent                " 依据上面的对齐格式，智能的选择对齐方式，对于类似C语言编写上有用   
set tabstop=4                  " 设置tab键为4个空格，
set showmatch                  " 设置匹配模式，类似当输入一个左括号时会匹配相应的右括号      
set guioptions-=T              " 去除vim的GUI版本中得toolbar   
set vb t_vb=                   " 当vim进行编辑时，如果命令错误，会发出警报，该设置去掉警报       
set ruler                      " 在编辑过程中，在右下角显示光标位置的状态行     
set incsearch

set rtp+=~/.fzf

source ~/.vim_runtime/vimrcs/basic.vim
source ~/.vim_runtime/vimrcs/filetypes.vim
source ~/.vim_runtime/vimrcs/plugins_config.vim
source ~/.vim_runtime/vimrcs/extended.vim

try
  source ~/.vim_runtime/my_configs.vim
catch
  endtry

call plug#begin('~/.vim/plugged')

" Make sure you use single quotes

" Shorthand notation; fetches https://github.com/junegunn/vim-easy-align
Plug 'junegunn/vim-easy-align'
Plug '~/.fzf'
" Any valid git URL is allowed
Plug 'https://github.com/junegunn/vim-github-dashboard.git'

" Multiple Plug commands can be written in a single line using | separators
Plug 'SirVer/ultisnips' | Plug 'honza/vim-snippets'

" On-demand loading
Plug 'scrooloose/nerdtree', { 'on':  'NERDTreeToggle' }
Plug 'tpope/vim-fireplace', { 'for': 'clojure' }

" Using a non-master branch
Plug 'rdnetto/YCM-Generator', { 'branch': 'stable' }

" Using a tagged release; wildcard allowed (requires git 1.9.2 or above)
Plug 'fatih/vim-go', { 'tag': '*' }

" Plugin options
Plug 'nsf/gocode', { 'tag': 'v.20150303', 'rtp': 'vim' }
Plug 'junegunn/fzf', { 'dir': '~/.fzf', 'do': './install --all' }
Plug '~/my-prototype-plugin'
Plug 'junegunn/goyo.vim'
Plug 'junegunn/seoul256.vim'
Plug 'lervag/vimtex'

call plug#end()

let g:tex_flavor='latex'
let g:vimtex_view_method='zathura'
let g:vimtex_quickfix_mode=0
set conceallevel=1
let g:tex_conceal='abdmg'
let g:go_version_warning = 0
let g:seoul256_background = 236
silent! colo seoul256