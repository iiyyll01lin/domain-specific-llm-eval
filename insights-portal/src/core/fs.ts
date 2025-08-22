export type DirHandle = FileSystemDirectoryHandle
export type FileHandle = FileSystemFileHandle

export async function* iterateDir(dir: DirHandle, depth = 0, maxDepth = 3): AsyncGenerator<{ path: string; handle: FileSystemHandle }> {
  if (depth > maxDepth) return
  for await (const [name, handle] of (dir as any).entries() as AsyncIterable<[string, FileSystemHandle]>) {
    const path = name
    yield { path, handle }
    if (handle.kind === 'directory') {
      for await (const sub of iterateDir(handle as DirHandle, depth + 1, maxDepth)) {
        (sub as any).path = `${name}/${sub.path}`
        yield sub
      }
    }
  }
}

export async function readFileToText(fileHandle: FileHandle) {
  const file = await fileHandle.getFile()
  return file.text()
}
