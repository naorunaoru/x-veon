export interface NewerRelease { name: string; version: string; url: string }
export interface UpdateHost { check(): Promise<NewerRelease | null> }
