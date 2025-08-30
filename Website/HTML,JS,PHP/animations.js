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
const sizeSelector = document.querySelector('.SizeSelector');
const tiles = document.querySelectorAll('.board > div');
const moveCount = document.querySelector('.moves');



function StartGame(){
    const noMoves = 0;
    const sizeSelect = document.getElementById('sizeSelect');
    selectedSize = parseInt(sizeSelect.value) || 3;
    

    //disapearences
    header.style.margin = '0';
    sizeSelector.style.opacity = '0';
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
    board.style.gridTemplateColumns = 'repeat('+selectedSize+', 1fr)';
    void board.offsetWidth;
    board.style.transform = 'scale(1)';
    timeC.style.transform = 'scale(1)';
    moves.style.transform = 'scale(1)';
    retry.style.transform = 'scale(1)';
    mlAssist.style.transform = 'scale(1)';
    setTimeout(createBoard(selectedSize), 500);
}

function toggleTiles(selectedSize) {
    const tiles = document.querySelectorAll('.board > div');

    tiles.forEach((tile) => {
            tile.style.display = 'flex';
            void tile.offsetWidth;
            tile.style.transform = 'scale(1)';

            if (selectedSize != 3 ) {
                tile.style.fontSize = '50px';/**he it swithces between 80px and 50px(look at the game.css) */
            }
        });
}

function createBoard(selectedSize){
    puzzle = createPuzzle(selectedSize);
    out = shuffle(puzzle,selectedSize);
    shuffledPuzzle = out[0];
    createTiles(shuffledPuzzle);
    toggleTiles(selectedSize);
}

function createPuzzle(selectedSize){
    let row = new Array(selectedSize).fill(0);
    let puzzle = new Array(selectedSize).fill(0);

    for(i = 0, j = 0, z = 0; i < (selectedSize*selectedSize); i++){
        if (j >= selectedSize-1) {
            row[j] = i+1;
            puzzle[z] = row;
            j=0;
            z++;
            row = new Array(selectedSize).fill(0);
        }
        else{
            row[j] = i+1;
            j++;
        }
    }
    //last index to 0
    puzzle[selectedSize-1][selectedSize-1]=0;
    return puzzle;
}

function createTiles(puzzle){
    let flatPuzzle = puzzle.flat()
    const fragment = document.createDocumentFragment();

    flatPuzzle.forEach((i) => {
        const div = document.createElement('div');
        if (i != 0) {
            div.className = 'tile';
            div.dataset.value = i
            div.textContent = `${i}`;
        }
        else{
            div.className = 'null';
        }

        fragment.appendChild(div);
    });

    board.innerHTML = '';
    board.appendChild(fragment);
}

function shuffle(puzzle,selectedSize){
    /**annalogy is UP means moving empty tile up */

    let max = 10000;
    let randomInt = Math.floor(Math.random() * max) + 1;
    CurrentIndex = findIndex2D(puzzle,0);
    row = CurrentIndex[0];
    col = CurrentIndex[1];
    let moves = [];

    for (let index = 0; index < randomInt; index++) {
        let validMoves = [];
        

        if (row > 0) validMoves.push('UP');
        if (row < selectedSize - 1) validMoves.push('DOWN');
        if (col > 0) validMoves.push('LEFT');
        if (col < selectedSize - 1) validMoves.push('RIGHT');

        // Choose a random valid move
        if (validMoves.length > 0) {
            let randomMove = validMoves[Math.floor(Math.random() * validMoves.length)];
            let newRow, newCol;
            switch (randomMove) {
                case 'UP':
                    newRow = row - 1;
                    newCol = col;
                    if (newRow >= 0) {
                        [puzzle[row][col], puzzle[newRow][newCol]] = [puzzle[newRow][newCol], puzzle[row][col]];
                    }
                    moves.push('UP');
                    break;
                case 'DOWN':
                    newRow = row + 1;
                    newCol = col;
                    if (newRow < selectedSize) {
                        [puzzle[row][col], puzzle[newRow][newCol]] = [puzzle[newRow][newCol], puzzle[row][col]];
                    }
                    moves.push('DOWN');
                    break;
                case 'RIGHT':
                    newRow = row;
                    newCol = col + 1;
                    if (newCol < selectedSize) {
                        [puzzle[row][col], puzzle[newRow][newCol]] = [puzzle[newRow][newCol], puzzle[row][col]];
                    }
                    moves.push('RIGHT');
                    break;
                case 'LEFT':
                    newRow = row;
                    newCol = col - 1;
                    if (newCol >= 0) {
                        [puzzle[row][col], puzzle[newRow][newCol]] = [puzzle[newRow][newCol], puzzle[row][col]];
                    }
                    moves.push('LEFT');
                    break;
                default:
                    break;
            }
            row = newRow;
            col = newCol;
        }
    }
    return [puzzle,moves];
}

function findIndex2D(array, target) {
    for (let row = 0; row < array.length; row++) {
        for (let col = 0; col < array[row].length; col++) {
            if (array[row][col] === target) {
                return [row, col]; // Return row and column indices
            }
        }
    }
    return [-1, -1]; // Return [-1, -1] if not found
}

function updateTileDOM(row, col, value) {
    // Calculate the index in the flat array
    const size = puzzle.length;
    const index = row * size + col;
    
    // Get the tile element
    const tile = document.querySelectorAll('.board > div')[index];
    
    if (value !== 0) {
        tile.className = 'tile';
        tile.dataset.value = value;
        tile.textContent = value;
    } else {
        tile.className = 'null';
        tile.dataset.value = '0';
        tile.textContent = '';
    }
}

function moveUP(puzzle) {
    let [row, col] = findIndex2D(puzzle, 0);
    let newRow = row - 1;
    let newCol = col;
    
    if (newRow >= 0) {
        // Store the value that will move into the empty space
        const movingValue = puzzle[newRow][newCol];
        
        // Swap in the array
        [puzzle[row][col], puzzle[newRow][newCol]] = [puzzle[newRow][newCol], puzzle[row][col]];
        
        // Update the DOM
        updateTileDOM(row, col, puzzle[row][col]); // Now contains the moved value
        updateTileDOM(newRow, newCol, puzzle[newRow][newCol]); // Now contains 0 (empty)
    }
}

function moveDOWN(puzzle) {
    let [row, col] = findIndex2D(puzzle, 0);
    const newRow = row + 1;
    const newCol = col;

    // Check if move is valid BEFORE accessing array
    if (newRow < puzzle.length) {
        [puzzle[row][col], puzzle[newRow][newCol]] = [puzzle[newRow][newCol], puzzle[row][col]];

        // Update the DOM
        updateTileDOM(row, col, puzzle[row][col]); // Now contains the moved value
        updateTileDOM(newRow, newCol, puzzle[newRow][newCol]); // Now contains 0 (empty)
    }

}

function moveRIGHT(puzzle){
    let [row, col] = findIndex2D(puzzle, 0);
    let newRow = row;
    let newCol = col + 1;
    if (newCol < puzzle.length) {
        [puzzle[row][col], puzzle[newRow][newCol]] = [puzzle[newRow][newCol], puzzle[row][col]];

        updateTileDOM(row, col, puzzle[row][col]); // Now contains the moved value
        updateTileDOM(newRow, newCol, puzzle[newRow][newCol]); // Now contains 0 (empty)
    }

}

function moveLEFT(puzzle){
    let [row, col] = findIndex2D(puzzle, 0);
    let newRow = row;
    let newCol = col - 1;
    if (newCol >= 0) {
        [puzzle[row][col], puzzle[newRow][newCol]] = [puzzle[newRow][newCol], puzzle[row][col]];

        updateTileDOM(row, col, puzzle[row][col]); // Now contains the moved value
        updateTileDOM(newRow, newCol, puzzle[newRow][newCol]); // Now contains 0 (empty)
    }
}

function availableMoves(puzzle) {
    let moves = [];
    let [row, col] = findIndex2D(puzzle, 0);
    
    // UP: row-1, col (check if row > 0)
    if (row > 0) moves.push(puzzle[row-1][col]);
    
    // DOWN: row+1, col (check if row < selectedSize-1)
    if (row < selectedSize-1) moves.push(puzzle[row+1][col]);
    
    // LEFT: row, col-1 (check if col > 0)
    if (col > 0) moves.push(puzzle[row][col-1]);
    
    // RIGHT: row, col+1 (check if col < selectedSize-1)
    if (col < selectedSize-1) moves.push(puzzle[row][col+1]);
    
    return moves;
}


const tile = document.querySelectorAll('.board > div.tile');
    const tileHover = document.querySelectorAll('.board > div.tile:hover');
    const tileActive = document.querySelectorAll('.board > div.tile:active');
    const nullTile = document.querySelectorAll('.board > div.null');

document.addEventListener('DOMContentLoaded', () => {
    const board = document.querySelector('.board');
    if (board) {
        board.addEventListener('mouseover', function(event){
            // Check if hovered element is a tile
            if (event.target.classList.contains('tile')) {
                const tileValue = event.target.dataset.value;
                const ablebMoves = availableMoves(shuffledPuzzle);
                console.log(ablebMoves);
                if (!ablebMoves.includes(parseInt(tileValue))) {
                    event.target.classList.add('tile-cant');
                }
            }
        });

        board.addEventListener('click', function(event) {
            // Check if clicked element is a tile
            if (event.target.classList.contains('tile')) {
                const tileValue = event.target.dataset.value;
                const [row, col] = findIndex2D(puzzle, parseInt(tileValue));
                const possibleMoves = availableMoves(shuffledPuzzle);
                let moved = false;

                if (possibleMoves.includes(parseInt(tileValue))) {
                    if (row > 0 && puzzle[row - 1][col] === 0) {
                        moveDOWN(shuffledPuzzle);
                        moved = true;
                    } else if (row < selectedSize - 1 && puzzle[row + 1][col] === 0) {
                        moveUP(shuffledPuzzle);
                        moved = true;
                    } else if (col > 0 && puzzle[row][col - 1] === 0) {
                        moveRIGHT(shuffledPuzzle);
                        moved = true;
                    } else if (col < selectedSize - 1 && puzzle[row][col + 1] === 0) {
                        moveLEFT(shuffledPuzzle);
                        moved = true;
                    }

                    if (moved) {
                        noMoves += 1;
                        moveCount.textContent = `Moves: ${noMoves}`;
                    }
                }


            }
        });
    }
});