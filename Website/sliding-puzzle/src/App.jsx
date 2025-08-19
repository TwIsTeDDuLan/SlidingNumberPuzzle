import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'

function App() {
  const [count, setCount] = useState(0)

  return (
    <>
      <div>
        <h1 className='text-3xl font-bold  text-blue-600'>
          Slide Puzzle!
        </h1>
        <dev class="flex gap-50 justify-center mt-5">
          <div>
            <form class="max-w-sm mx-auto">
              <label for="countries" class="block mb-2 text-sm font-medium text-gray-900 dark:text-white">Select an option</label>
              <select id="countries" class="
              bg-gray-50 border border-gray-300 text-gray-900 text-sm text-center rounded-lg 
              focus:ring-blue-500 focus:border-blue-500 block w-full p-2.5 dark:bg-gray-700 
              dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 
              dark:focus:border-blue-500">
                <option selected>Select Board</option>
                <option value="3">3X3</option>
                <option value="4">4X4</option>
                <option value="5">5X5</option>
              </select>
            </form>
          </div>
        </dev>

      </div>  

      <div className="relative flex justify-center items-center mt-10">
        {/* Aura effect */}
        <div className="absolute inset-1 animate-ping rounded-3xl bg-blue-500 opacity-75" style={{ animationDuration: '3s' }}></div>
        
        {/* Main button */}
        <button className='
          relative
          w-60 h-30 
          rounded-3xl border border-transparent
          px-6 py-4 
          text-5xl font-medium
          bg-gray-900 
          cursor-pointer 
          transition-all duration-250 
          hover:border-gray-300
          flex items-center justify-center
          text-white
          z-10  /* Ensure button is above aura */
        '>
          Start
        </button>
      </div>
    </>
  )
}

export default App

