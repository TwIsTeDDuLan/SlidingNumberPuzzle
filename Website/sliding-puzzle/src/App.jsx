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
        <p className='text-lg text-gray-600'>
          A fun and interactive sliding puzzle game built with React and Vite.
        </p>
        <dev class="flex gap-50 justify-center mt-20">
          <div class="flex items-center">
            <input type="radio" id="option1" name="myRadioGroup" value="option1" class="form-radio text-blue-1000  w-8 h-8"/>
            <label for="option1" class="ml-2 text-gray-1000 text-4xl text-">3x3</label>
          </div>

          <div class="flex items-center">
            <input type="radio" id="option2" name="myRadioGroup" value="option2" class="form-radio text-blue-1000 w-4 h-4"/>
            <label for="option2" class="ml-2 text-gray-1000">4x4</label>
          </div>
        </dev>

      </div>  

      <div className='flex justify-center align-center'>
            <button className='w-30 h-20'>Start</button>
      </div>
    </>
  )
}

export default App

