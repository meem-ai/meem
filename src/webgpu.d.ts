declare module "webgpu" {
  export const globals: Record<string, unknown>
  export const create: (options: string[]) => GPU
}
