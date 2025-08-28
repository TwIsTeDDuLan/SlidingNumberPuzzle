//elements

const playArea = document.querySelector('.play-area');
const stat = document.querySelector('.stats');
const mlAssist = document.querySelector('.ml-assist');
const timeC = document.querySelector('.time');
const moves = document.querySelector('.moves');
const retry = document.querySelector('.retry');
const board = document.querySelector('.board');/*board > div here as well*/

const header = document.querySelector('.header');/*go up*/

const controls = document.querySelector('.controls');
const startBtnCls = document.querySelector('.aura');
const startBtn = document.getElementById('start');
const sizeSelcetor = document.querySelector('.SizeSelector');
let size = document.getElementById('sizeSelect').value;/*one arrasize */
const tiles = document.querySelectorAll('.board > div');



function StartGame(){
    //disapearences
    header.style.margin = '0';
    sizeSelcetor.style.opacity = '0';
    startBtnCls.style.display = 'none';
    controls.style.display = 'none';

    // apearings
    playArea.style.display = 'grid';
    stat.style.display = 'flex';
    mlAssist.style.display = 'flex';
    timeC.style.display = 'flex';
    moves.style.display = 'flex';
    retry.style.display = 'flex';
    board.style.display = 'grid';
    void board.offsetWidth;
    board.style.transform = 'scale(1)';
    timeC.style.transform = 'scale(1)';
    moves.style.transform = 'scale(1)';
    retry.style.transform = 'scale(1)';
    mlAssist.style.transform = 'scale(1)';
    setTimeout(createBoard, 500);
}

function toggleTiles() {
    const tiles = document.querySelectorAll('.board > div');

    tiles.forEach((tile) => {
            tile.style.display = 'flex';
            void tile.offsetWidth;
            tile.style.transform = 'scale(1)';
        });
}

function createBoard(){
    let row = new Array(size).fill(0);
    let puzzle = new Array(size).fill(0);

    for(i = 0, j = 0, z = 0; i < (size*size)-1; i++){
        if(j >= 2){
            j=0;
            puzzle[z] = row;
            z++;
        }
        row[i] = i+1;
        j++;
    }
    console.log(puzzle);
}