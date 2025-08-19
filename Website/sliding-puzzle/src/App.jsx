import { useState } from 'react';
import './App.css'

function App() {
  const [gameStarted, setGameStart] = useState(false);
  const [isStarting, setIsStarting] = useState(false);
  const [size, setSize] = useState(3);

  const handleStart = () => {
    setIsStarting(true);
    console.log("HandleStart");
    setTimeout(() => {
      setGameStart(true);
      setIsStarting(false);
    }, 1000);
  };

  if(gameStarted) {
    return <Game boardSize={size} />
  }

  return (
    <>
      <div className={ `${isStarting ? 'mt-0 transition-all duration-1000' : 'mt-[20vh]'}`}>
        <h1 className={'text-3xl font-bold  text-blue-400'}>
            Slide Puzzle!
        </h1>
          <div className={`transition-opacity duration-500
            ${isStarting ? 'opacity-0' : 'opacity-100'}`}>
            <div className={`flex gap-50 justify-center mt-5`}>
              <div>
                <form class="max-w-sm mx-auto">
                  <label for="countries" class="block mb-2 text-sm font-medium text-gray-900 dark:text-white">Select an option</label>
                  <select id="countries" class="
                  bg-gray-50 border border-gray-300 text-gray-900 text-sm text-center rounded-lg 
                  focus:ring-blue-500 focus:border-blue-500 block w-full p-2.5 dark:bg-gray-700 
                  dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 
                  dark:focus:border-blue-500"
                  onChange={(event) => setSize(event.target.value)}>
                    <option selected>Select Board</option>
                    <option value="3">3X3</option>
                    <option value="4">4X4</option>
                    <option value="5">5X5</option>
                  </select>
                </form>
              </div>
            </div> 
          <div className="relative flex justify-center items-center mt-10">
            {/* Aura effect */}
            <div className="absolute inset-1 animate-ping rounded-3xl bg-blue-500 opacity-75" style={{ animationDuration: '3s' }}></div>
            
            {/* Main button */}
            <button className='
              relative
              w-55 h-30 
              rounded-4xl border border-transparent
              px-6 py-4 
              text-5xl font-medium
              bg-blue-500 
              cursor-pointer 
              transition-all duration-250 
              hover:border-gray-300
              flex items-center justify-center
              text-white
              z-10  /* Ensure button is above aura */
            '
            onClick={handleStart}
            disabled={isStarting}>
              {isStarting ? 'Loading...' : 'Start'}
            </button>
          </div>
        </div>
      </div>  
    </>
  );
}

function Game({boardSize}) {

  return(
    <>
      <h1 className={'text-3xl font-bold  text-blue-400'}>
            Slide Puzzle!
      </h1>
      <div className='
      bg-blue-500
      w-100 h-100
      rounded-4xl'>
          
      </div>
    </>
  );
}

export default App

