import { useEffect, useMemo, useState } from 'react';
import './App.css'

function App() {
  const [gameStarted, setGameStart] = useState(false);
  const [isStarting, setIsStarting] = useState(false);
  const [selectedSize, setSelectedSize] = useState(3);
  const [moves, setMoves] = useState(0);
  const [seconds, setSeconds] = useState(0);

  const handleStart = () => {
    setIsStarting(true);
    setTimeout(() => {
      setGameStart(true);
      setIsStarting(false);
      setMoves(0);
      setSeconds(0);
    }, 500);
  };

  const handleRetry = () => {
    setMoves(0);
    setSeconds(0);
  };

  if (gameStarted) {
    return (
      <Game
        selectedSize={selectedSize}
        onMove={() => setMoves((m) => m + 1)}
        moves={moves}
        seconds={seconds}
        setSeconds={setSeconds}
        onRetry={handleRetry}
      />
    );
  }

  return (
    <>
      <div className={`${isStarting ? 'mt-0 transition-all duration-1000' : 'mt-[20vh]'}`}>
        <div className={'text-3xl font-bold  text-blue-400'}>
          SlidingPuzzle
        </div>
        <div className={`transition-opacity duration-500 ${isStarting ? 'opacity-0' : 'opacity-100'}`}>
          <div className={`flex gap-50 justify-center mt-5`}>
            <div>
              <label className="block mb-2 text-sm font-medium">Select Size :</label>
              <select
                className="bg-gray-50 border border-gray-300 text-gray-900 text-sm text-center rounded-lg p-2.5 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                value={selectedSize}
                onChange={(e) => setSelectedSize(parseInt(e.target.value) || 3)}
              >
                <option value={3}>3x3</option>
                <option value={4}>4x4</option>
                <option value={5}>5x5</option>
                <option value={6}>6x6</option>
              </select>
            </div>
          </div>
          <div className="relative flex justify-center items-center mt-10">
            <div className="absolute inset-1 animate-ping rounded-3xl bg-blue-500 opacity-75" style={{ animationDuration: '3s' }}></div>
            <button
              className='relative w-55 h-30 rounded-4xl border border-transparent px-6 py-4 text-5xl font-medium bg-blue-500 cursor-pointer transition-all duration-250 hover:border-gray-300 flex items-center justify-center text-white z-10'
              onClick={handleStart}
              disabled={isStarting}
            >
              {isStarting ? 'Loading...' : 'START'}
            </button>
          </div>
        </div>
      </div>
    </>
  );
}

function Game({ selectedSize, onMove, moves, seconds, setSeconds, onRetry }) {
  const [shuffledPuzzle, setShuffledPuzzle] = useState(() => shuffle(createPuzzle(selectedSize), selectedSize)[0]);

  // restart when size changes
  useEffect(() => {
    const base = createPuzzle(selectedSize);
    const [shuffled] = shuffle(deepClone(base), selectedSize);
    setShuffledPuzzle(shuffled);
  }, [selectedSize]);

  // simple timer
  useEffect(() => {
    const id = setInterval(() => setSeconds((s) => s + 1), 1000);
    return () => clearInterval(id);
  }, [setSeconds]);

  const timeText = useMemo(() => {
    const m = Math.floor(seconds / 60).toString().padStart(2, '0');
    const s = (seconds % 60).toString().padStart(2, '0');
    return `${m}:${s}`;
  }, [seconds]);

  const handleTileClick = (value) => {
    if (value === 0) return;
    const [row, col] = findIndex2D(shuffledPuzzle, value);
    const [zr, zc] = findIndex2D(shuffledPuzzle, 0);

    // must be adjacent
    const isAdjacent = (Math.abs(row - zr) + Math.abs(col - zc)) === 1;
    if (!isAdjacent) return;

    const next = structuredClone(shuffledPuzzle);
    [next[zr][zc], next[row][col]] = [next[row][col], next[zr][zc]];
    setShuffledPuzzle(next);
    onMove();
  };

  const ableToMove = useMemo(() => {
    const values = new Set();
    const [zr, zc] = findIndex2D(shuffledPuzzle, 0);
    if (zr > 0) values.add(shuffledPuzzle[zr - 1][zc]);
    if (zr < selectedSize - 1) values.add(shuffledPuzzle[zr + 1][zc]);
    if (zc > 0) values.add(shuffledPuzzle[zr][zc - 1]);
    if (zc < selectedSize - 1) values.add(shuffledPuzzle[zr][zc + 1]);
    return values;
  }, [shuffledPuzzle, selectedSize]);

  const retry = () => {
    const base = createPuzzle(selectedSize);
    const [shuffled] = shuffle(deepClone(base), selectedSize);
    setShuffledPuzzle(shuffled);
    onRetry();
  };

  return (
    <>
      <div className="container">
        <div className="header"><h1>SlidingPuzzle</h1></div>
        <div className="play-area">
          <div className="stats">
            <div className="time">Time<br />{timeText}</div>
            <div className="best-time">Your Best Time<br />00:00</div>
            <div className="moves">Moves<br />{moves}</div>
            <div className="best-moves">Your Best Moves:<br />0</div>
            <button className="retry" onClick={retry}>Retry?</button>
          </div>
          <div
            className="board"
            style={{ gridTemplateColumns: `repeat(${selectedSize}, 1fr)` }}
          >
            {shuffledPuzzle.flat().map((val, idx) => {
              const isNull = val === 0;
              const canMove = ableToMove.has(val);
              const fontSize = selectedSize !== 3 ? '50px' : '80px';
              return (
                <div
                  key={idx}
                  className={`${isNull ? 'null' : 'tile'} ${!isNull && !canMove ? 'tile-cant' : ''}`}
                  data-value={val}
                  onClick={() => handleTileClick(val)}
                  style={{ fontSize }}
                >
                  {isNull ? '' : val}
                </div>
              );
            })}
          </div>
          <div className="ml-assist">ML Suggestions</div>
        </div>
      </div>
      <div className="footer">
          <p>© 2025 SlidingPuzzle, Twisted (Pvt) Ltd. All rights reserved.</p>
      </div>
    </>
  );
}

// ----- Logic ported from animations.js (adapted to React state) -----
function createPuzzle(selectedSize) {
  const puzzle = Array.from({ length: selectedSize }, () => Array.from({ length: selectedSize }, () => 0));
  let value = 1;
  for (let r = 0; r < selectedSize; r++) {
    for (let c = 0; c < selectedSize; c++) {
      puzzle[r][c] = value;
      value += 1;
    }
  }
  puzzle[selectedSize - 1][selectedSize - 1] = 0;
  return puzzle;
}

function shuffle(puzzle, selectedSize) {
  let [row, col] = findIndex2D(puzzle, 0);
  const max = 10000;
  const randomInt = Math.floor(Math.random() * max) + 1;
  for (let i = 0; i < randomInt; i++) {
    const validMoves = [];
    if (row > 0) validMoves.push('UP');
    if (row < selectedSize - 1) validMoves.push('DOWN');
    if (col > 0) validMoves.push('LEFT');
    if (col < selectedSize - 1) validMoves.push('RIGHT');
    if (validMoves.length === 0) continue;
    const mv = validMoves[Math.floor(Math.random() * validMoves.length)];
    let newRow = row;
    let newCol = col;
    if (mv === 'UP') newRow = row - 1;
    if (mv === 'DOWN') newRow = row + 1;
    if (mv === 'LEFT') newCol = col - 1;
    if (mv === 'RIGHT') newCol = col + 1;
    [puzzle[row][col], puzzle[newRow][newCol]] = [puzzle[newRow][newCol], puzzle[row][col]];
    row = newRow; col = newCol;
  }
  return [puzzle, []];
}

function findIndex2D(array2d, target) {
  for (let r = 0; r < array2d.length; r++) {
    for (let c = 0; c < array2d[r].length; c++) {
      if (array2d[r][c] === target) return [r, c];
    }
  }
  return [-1, -1];
}

function deepClone(matrix) {
  return matrix.map((row) => row.slice());
}

export default App

