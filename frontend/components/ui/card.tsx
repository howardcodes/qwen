import clsx from 'clsx'
import { HTMLAttributes } from 'react'

export function Card({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return <div className={clsx('rounded-3xl border border-slate-200/70 bg-white/90 p-6 shadow-sm shadow-slate-200/60 backdrop-blur', className)} {...props} />
}
