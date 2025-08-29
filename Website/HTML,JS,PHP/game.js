import { shuffledPuzzle } from "./animations.js";

console.log(shuffledPuzzle);

function moveUP(row, col, puzzle){
    const newRow = row - 1;
    const newCol = col;
    if (newRow >= 0) {
        [puzzle[row][col], puzzle[newRow][newCol]] = [puzzle[newRow][newCol], puzzle[row][col]];
    }

    const upTile = document.querySelector(`.tile[data-value='${puzzle[newRow][newCol]}']`);
    upTile.style.backgroundColor = 'red';
}

function moveDOWN(){

}

function moveRIGHT(){

}

function moveLEFT(){
    
}