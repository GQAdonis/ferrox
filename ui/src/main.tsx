import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import {
  createBrowserRouter,
  Navigate,
  RouterProvider,
} from "react-router";
import { AppShell } from "@/components/app-shell";
import { ChatScreen } from "@/screens/chat";
import { ModelsScreen } from "@/screens/models";
import { ActivityScreen } from "@/screens/activity";
import { ConnectScreen } from "@/screens/connect";
import "@/index.css";

// `/` and `/ui` both land on Chat; `/ui/<screen>` deep links resolve
// directly. The server answers every one of these paths with the same
// shell (see `crates/ferrox-server/src/ui.rs`), so a reload or a
// bookmark lands on the right screen instead of a 404.
const router = createBrowserRouter([
  {
    path: "/",
    element: <AppShell />,
    children: [
      { index: true, element: <Navigate to="/ui/chat" replace /> },
      { path: "ui", element: <Navigate to="/ui/chat" replace /> },
      { path: "ui/chat", element: <ChatScreen /> },
      { path: "ui/models", element: <ModelsScreen /> },
      { path: "ui/activity", element: <ActivityScreen /> },
      { path: "ui/connect", element: <ConnectScreen /> },
      // Anything else the SPA fallback handed us is not a screen.
      { path: "*", element: <Navigate to="/ui/chat" replace /> },
    ],
  },
]);

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <RouterProvider router={router} />
  </StrictMode>,
);
