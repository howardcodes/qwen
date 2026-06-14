import clsx from 'clsx'
import { HTMLAttributes } from 'react'

export function Card({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return <div className={clsx('rounded-2xl border border-white/10 bg-card/80 p-5 shadow-xl', className)} {...props} />
}
