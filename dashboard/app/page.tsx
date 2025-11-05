import { Button } from "@/components/ui/button";
import Link from "next/link";

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-24">
      <div className="z-10 w-full max-w-5xl items-center justify-between font-mono text-sm lg:flex">
        <h1 className="text-4xl font-bold">
          Welcome to the CASE STUDY DASHBOARD
        </h1>

        <Button>
          <Link href="/dashboard">
          Get Started
          </Link>
        </Button>
      </div>
    </main>
  );
}
