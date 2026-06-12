import clsx from 'clsx'
import { ButtonHTMLAttributes } from 'react'

export function Button({ className, ...props }: ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      className={clsx(
        'rounded-xl bg-primary px-4 py-2 font-semibold text-white shadow-lg shadow-primary/30 transition hover:scale-[1.02] disabled:cursor-not-allowed disabled:opacity-60',
        className
      )}
      {...props}
    />
  )
}
